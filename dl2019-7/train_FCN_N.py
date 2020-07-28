from __future__ import print_function
import glob
import os
import gzip
from datetime import datetime
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent.bc_agent import BCAgent
from config import Config
from utils import rgb2gray

# bug in cdnn: https://github.com/pytorch/pytorch/issues/4107
torch.backends.cudnn.enabled = False


def read_data_generator(datasets_dir="./data", frac=0.1, is_fcn=True):
    print(f'\nreading  data from {datasets_dir}')
    file_names = glob.glob(os.path.join(datasets_dir, "*.gzip"))
    print(f'files: {file_names}')

    for data_file in file_names:
        print(f'\n--- current file: {data_file} ---')
        X = []
        y = []
        f = gzip.open(data_file, 'rb')
        data = pickle.load(f)
        n_samples = len(data["state"])
        # Hint: to access images use state_img here!
        if is_fcn:
            X.extend(data["state"])
        else:
            X.extend(data["state_img"])

        y.extend(data["action"])

        X = np.array(X).astype("float32")
        y = np.array(y).astype("float32")

        # split data into training and validation set
        X_train, y_train = (
            X[:int((1 - frac) * n_samples)],
            y[:int((1 - frac) * n_samples)],
        )
        X_valid, y_valid = (
            X[int((1 - frac) * n_samples):],
            y[int((1 - frac) * n_samples):],
        )
        print(f"X_train shape: {np.shape(X_train)}, \
            y_train shape: {np.shape(y_train)}")
        print(f"X_valid shape: {np.shape(X_valid)}, \
            y_valid shape: {np.shape(y_valid)}")

        yield X_train, y_train, X_valid, y_valid


def skip_frames_FCN(input_data, input_label, skip_no, history_length):
    assert skip_no > 0, f'config.skip_n: {skip_no}. Must be > 0'
    N_train = input_data.shape[0]
    skipped_data = []
    skipped_label = []
    skip_pointer = 0
    while skip_pointer < N_train:
        data = np.dstack(input_data[skip_pointer:skip_pointer + history_length, ...])
        data = np.swapaxes(data, 0, 2)
        label = input_label[skip_pointer:skip_pointer + history_length, ...]
        skipped_data.append(data)
        skipped_label.append(label)
        skip_pointer += skip_no + history_length
    skipped_data = np.array(skipped_data)
    skipped_label = np.array(skipped_label)
    return skipped_data, skipped_label


def skip_frames(input_data, input_label, skip_no, history_length):
    assert skip_no > 0, f'config.skip_n: {skip_no}. Must be > 0'
    N_train = input_data.shape[0]
    skipped_data = []
    skipped_label = []
    skip_pointer = history_length
    while skip_pointer < N_train:
        data = np.dstack(input_data[skip_pointer -
                                    history_length:skip_pointer + 1, ...])
        label = input_label[skip_pointer, ...]
        skipped_data.append(data)
        skipped_label.append(label)
        skip_pointer += skip_no
    skipped_data = np.array(skipped_data)
    skipped_label = np.array(skipped_label)
    return skipped_data, skipped_label


def preprocessing(X_train, y_train, X_valid, y_valid, conf):
    # --- preprocessing for state vector ---
    if conf.is_fcn:
        X_train, y_train = skip_frames_FCN(input_data=X_train,
                                       input_label=y_train,
                                       skip_no=conf.skip_frames,
                                       history_length=conf.history_length)
        X_train = X_train.squeeze()
        y_train = y_train.squeeze()
        X_shape = X_train.shape
        y_shape = y_train.shape
        X_train = np.reshape(X_train, (X_shape[0] * X_shape[1], X_shape[2]))
        y_train = np.reshape(y_train, (y_shape[0] * y_shape[1], 1))
        return X_train, y_train, X_valid, y_valid

    # --- preprocessing for image data ---
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (100, 150, 1)
    # X_train shape: (N_sample, H, W, 3)
    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (200, 300, 1). Later, add a history of the last N images to your state so that a state has shape (200, 300, N).

    # Hint: you can also implement frame skipping
    # skip_frames: parameter similar to stride in CNN
    skip_n = conf.skip_frames
    # history_length: images representing the history in each data point.
    hist_len = conf.history_length
    # X_train shape: (2250, 200, 300, 3)
    X_train, y_train = skip_frames(input_data=X_train,
                                   input_label=y_train,
                                   skip_no=skip_n,
                                   history_length=hist_len)

    X_valid, y_valid = skip_frames(input_data=X_valid,
                                   input_label=y_valid,
                                   skip_no=skip_n,
                                   history_length=hist_len)

    return X_train, y_train, X_valid, y_valid


def create_data_loader(X, y, batch_size, shuffle=True):
    """
    Creates pytorch dataloaders from numpy data and labels
    """
    tensor_X = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=2)
    return loader


def calculate_weights(labels):
    """
    function for calculating weights for each class in unbalanced datasets
    """
    classes = np.unique(labels)
    counts = np.zeros_like(classes)
    for i, c in enumerate(classes):
        counts[i] = np.count_nonzero(labels == c)
    weights = 1.0 / counts
    weights = weights / np.linalg.norm(weights, ord=1)  # normalization
    # side_engine_weights = (weights[1] +
    #                        weights[3]) / 2  # same weights for side engines
    # weights[1] = weights[3] = side_engine_weights
    return torch.Tensor(weights).view(-1, 1)


def train_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
    config,
    exp_timestamp=None,
    global_step=None,
    model_dir="./models",
    tensorboard_dir="./tensorboard",
):
    n_epochs = config.n_epochs
    batch_size = config.batch_size
    tb_saving_res = 2
    timestamp = exp_timestamp if exp_timestamp else datetime.now().strftime(
        "%Y-%m-%d--%H-%M")
    modelfilename = os.path.join(model_dir, f'agent_{timestamp}.pt')
    step = global_step if global_step else 0

    # --- create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # --- tensorboard writer
    writer = SummaryWriter(log_dir=f'{tensorboard_dir}/experiment_{timestamp}')
    # --- agent ---
    weights_classes = calculate_weights(labels=y_train)
    print(f'class weights: {weights_classes.squeeze()}')
    agent = BCAgent(config, weights=weights_classes)
    if exp_timestamp:  # start form previous save
        agent.load(modelfilename, to_cpu=False)
        print(f'loaded model: {modelfilename}')
    agent.to_device()
    agent.train_mode()
    # --- dataloaders ---
    trainloader = create_data_loader(X_train, y_train, batch_size=batch_size)
    validloader = create_data_loader(X_valid,
                                     y_valid,
                                     batch_size=X_valid.shape[0])
    # --- training loop ---
    training_loss = 0.0
    for epoch in range(n_epochs):
        print(f'\nEpoch:{epoch}')
        for i, (X, y) in tqdm(enumerate(trainloader), desc='Batch Training '):
            loss = agent.update(X, y)
            training_loss += loss.item()
            if (i + 1) % tb_saving_res == 0:
                step += tb_saving_res
                # --- record training loss ---
                writer.add_scalar(tag='training_loss',
                                  scalar_value=(training_loss / tb_saving_res),
                                  global_step=step)
                training_loss = 0.0
                # --- record validation loss ---
                agent.test_mode()
                x_val, y_val = next(iter(validloader))
                y_pred = agent.predict(X=x_val).detach()
                val_loss = agent.prediction_loss(y_pred=y_pred, y=y_val)
                writer.add_scalar(tag='validation_loss',
                                  scalar_value=val_loss,
                                  global_step=step)
                agent.train_mode()

    writer.close()

    # --- delete loaded data from RAM ---
    del X_train
    del y_train
    del X_valid
    del y_valid
    # --- save your agent
    agent.save(modelfilename)
    print(f'Model saved in file: {modelfilename}')
    return timestamp, step


if __name__ == "__main__":
    # read data
    conf = Config()
    is_fcn = conf.is_fcn
    experiment_timestamp = None
    global_step = None
    for training_data in read_data_generator("./data", is_fcn=is_fcn):
        X_train, y_train, X_valid, y_valid = training_data

        # preprocess data
        print('\n--- preprocessing data ---')
        X_train, y_train, X_valid, y_valid = preprocessing(
            X_train, y_train, X_valid, y_valid, conf)

        print('\n--- training ---')
        # train model (you can change the parameters!)
        experiment_timestamp, global_step = train_model(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            config=conf,
            global_step=global_step,
            exp_timestamp=experiment_timestamp)
