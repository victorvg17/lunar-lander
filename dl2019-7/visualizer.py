from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from train import read_labels_generator
sns.set()


class Visualizer:
    def __init__(self, result_path, showplots=True):
        self._result_path = result_path
        self._showplots = showplots

    def plot(self,
             data,
             title,
             xlabel,
             ylabel,
             legend,
             plot_name,
             smoothing=20):
        """
        data: list of tuples (list,list)
        legend: list of strings
        """

        fig, ax = plt.subplots()
        colors = [
            'indigo', 'limegreen', 'darkorange', 'dimgray', 'navy', 'olive',
            'blueviolet', 'darkred', 'midnightblue'
        ]
        cset = np.random.choice(colors, replace=False, size=len(data))
        for i, (x, y) in enumerate(data):
            sns.lineplot(x=x, y=y, ax=ax, alpha=0.2, color=cset[i])
            sns.lineplot(x,
                         pd.Series(y).rolling(window=smoothing).mean(),
                         ax=ax,
                         color=cset[i],
                         label=legend[i])
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        Path.mkdir(self._result_path, parents=True, exist_ok=True)
        fig.savefig(self._result_path / f'{plot_name}.png')
        if self._showplots:
            plt.show()
        else:
            plt.close()

    def plot_actions_histogram(self, 
                            datasets_dir,
                            title,
                            xlabel,
                            ylabel,
                            legend,
                            plot_name):
        """
        function for plotting histogram of action distributions.
        """
        data_generator = read_labels_generator(datasets_dir=datasets_dir)
        actions_holder = []
        for labels in data_generator:
            actions_holder.append(labels)
        actions_holder = np.array(actions_holder)
        actions_holder = actions_holder.flatten()
        print(f'total number of actions: {actions_holder.shape}')

        fig, ax = plt.subplots()
        sns.distplot(actions_holder, 
                    bins=4, 
                    kde=False,
                    color="darkblue",
                    ax=ax, 
                    label=legend, 
                    hist_kws = {'align' : 'mid', 'rwidth' : 0.7})
        for i, p in enumerate(ax.patches):
            ax.text(x=p.get_x() + p.get_width()/2.,
                    y=p.get_height(),
                    s=legend[i],
                    fontsize=14,
                    color='darkslategray',
                    ha='center',
                    va='bottom')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks([])


        Path.mkdir(self._result_path, parents=True, exist_ok=True)
        fig.savefig(self._result_path / f'{plot_name}.png')
        if self._showplots:
            plt.show()
        else:
            plt.close()

if __name__ == "__main__":
    plot_path = Path(__file__).resolve().parent / 'plots'
    viz = Visualizer(result_path=plot_path, showplots=True)

    # --- FCN Batchnorm Effect ---
    if False:
        data_no_batch_norm = pd.read_csv(
            "plot_data/experiments/fcn_batchnorm_effect/2020-03-04--17-08-tag-training_loss_no_batchnorm.csv"
        )
        data_with_batch_norm = pd.read_csv(
            "plot_data/experiments/fcn_batchnorm_effect/2020-03-04--17-15-tag-training_loss_with_batchnorm.csv"
        )
        x_nbn = data_no_batch_norm['Step'].tolist()
        y_nbn = data_no_batch_norm['Value'].tolist()
        x_bn = data_with_batch_norm['Step'].tolist()
        y_bn = data_with_batch_norm['Value'].tolist()
        data = [(x_nbn, y_nbn), (x_bn, y_bn)]
        legend = ['no Batch Norm', 'with Batch Norm']
        viz.plot(data=data,
                 title='Effect of Batch normalization on Fully Connected NN',
                 xlabel='steps',
                 ylabel='loss',
                 legend=legend,
                 plot_name='fcn_batch_norm_effect')

    # --- FCN Network Size ---
    if False:
        fcn_256 = pd.read_csv(
            "plot_data/experiments/fcn_network_size/2020-03-04--18-03-tag-training_loss_net_256.csv"
        )
        fcn_128 = pd.read_csv(
            "plot_data/experiments/fcn_network_size/2020-03-04--17-56-tag-training_loss_net_128.csv"
        )
        fcn_64 = pd.read_csv(
            "plot_data/experiments/fcn_network_size/2020-03-04--17-50-tag-training_loss_net_64.csv"
        )
        fcn_32 = pd.read_csv(
            "plot_data/experiments/fcn_network_size/2020-03-04--17-26-tag-training_loss_net_32.csv"
        )
        fcn_16 = pd.read_csv(
            "plot_data/experiments/fcn_network_size/2020-03-04--17-46-tag-training_loss_net_16.csv"
        )

        x_256 = fcn_256['Step'].tolist()
        y_256 = fcn_256['Value'].tolist()
        x_128 = fcn_128['Step'].tolist()
        y_128 = fcn_128['Value'].tolist()
        x_64 = fcn_64['Step'].tolist()
        y_64 = fcn_64['Value'].tolist()
        x_32 = fcn_32['Step'].tolist()
        y_32 = fcn_32['Value'].tolist()
        x_16 = fcn_16['Step'].tolist()
        y_16 = fcn_16['Value'].tolist()

        data = [(x_256, y_256), (x_128, y_128), (x_64, y_64), (x_32, y_32),
                (x_16, y_16)]
        legend = ['fcn 256', 'fcn 128', 'fcn 64', 'fcn 32', 'fcn 16']
        viz.plot(data=data,
                 title='Effect of Fully Connected NN Size',
                 xlabel='steps',
                 ylabel='loss',
                 legend=legend,
                 plot_name='fcn_net_size',
                 smoothing=5)

# --- FCN Dropout Effect training loss ---
if False:
    fcn_with = pd.read_csv(
        "plot_data/experiments/fcn_dropout_effect/2020-03-04--19-25-tag-training_loss_with_dropout_all_data.csv"
    )
    fcn_without = pd.read_csv(
        "plot_data/experiments/fcn_dropout_effect/2020-03-04--18-47-tag-training_loss_no_dropout_all_data.csv"
    )

    x_with = fcn_with['Step'].tolist()
    y_with = fcn_with['Value'].tolist()
    x_without = fcn_without['Step'].tolist()
    y_without = fcn_without['Value'].tolist()

    data = [(x_with, y_with), (x_without, y_without)]
    legend = ['with Dropout', 'without Dropout']
    viz.plot(data=data,
             title='Effect of Dropout on Training Loss',
             xlabel='steps',
             ylabel='loss',
             legend=legend,
             plot_name='fcn_training_dropout_effect')

# --- FCN Dropout effect validation loss ---
if False:
    fcn_with = pd.read_csv(
        "plot_data/experiments/fcn_dropout_effect/2020-03-04--19-25-tag-validation_loss_with_dropout_all_data.csv"
    )
    fcn_without = pd.read_csv(
        "plot_data/experiments/fcn_dropout_effect/2020-03-04--18-47-tag-validation_loss_no_dropout_all_data.csv"
    )

    x_with = fcn_with['Step'].tolist()
    y_with = fcn_with['Value'].tolist()
    x_without = fcn_without['Step'].tolist()
    y_without = fcn_without['Value'].tolist()

    data = [(x_with, y_with), (x_without, y_without)]
    legend = ['with Dropout', 'without Dropout']
    viz.plot(data=data,
             title='Effect of Dropout on Validation Loss',
             xlabel='steps',
             ylabel='loss',
             legend=legend,
             plot_name='fcn_validation_dropout_effect')

# --- FCN Different LR Training ---
if False:
    fcn_lr_5 = pd.read_csv(
        "plot_data/experiments/fcn_lr/2020-03-07--21-04-tag-training_loss_lr_5.csv"
    )
    fcn_lr_4 = pd.read_csv(
        "plot_data/experiments/fcn_lr/2020-03-07--21-26-tag-training_loss_lr_4.csv"
    )
    fcn_lr_3 = pd.read_csv(
        "plot_data/experiments/fcn_lr/2020-03-07--21-41-tag-training_loss_lr_3.csv"
    )

    x_5 = fcn_lr_5['Step'].tolist()[0:680]
    y_5 = fcn_lr_5['Value'].tolist()[0:680]
    x_4 = fcn_lr_4['Step'].tolist()
    y_4 = fcn_lr_4['Value'].tolist()
    x_3 = fcn_lr_3['Step'].tolist()
    y_3 = fcn_lr_3['Value'].tolist()

    data = [(x_5, y_5), (x_4, y_4), (x_3, y_3)]
    legend = ['LR 1e-5', 'LR 1e-4', 'LR 1e-3']
    viz.plot(data=data,
             title='Effect of Learning Rate on Training Loss',
             xlabel='steps',
             ylabel='loss',
             legend=legend,
             plot_name='fcn_training_lr_effect',
             smoothing=20)

# --- FCN Different LR Validation ---
if False:
    fcn_lr_5 = pd.read_csv(
        "plot_data/experiments/fcn_lr/2020-03-07--21-04-tag-validation_loss_lr_5.csv"
    )
    fcn_lr_4 = pd.read_csv(
        "plot_data/experiments/fcn_lr/2020-03-07--21-26-tag-validation_loss_lr_4.csv"
    )
    fcn_lr_3 = pd.read_csv(
        "plot_data/experiments/fcn_lr/2020-03-07--21-41-tag-validation_loss_lr_3.csv"
    )

    x_5 = fcn_lr_5['Step'].tolist()[0:680]
    y_5 = fcn_lr_5['Value'].tolist()[0:680]
    x_4 = fcn_lr_4['Step'].tolist()
    y_4 = fcn_lr_4['Value'].tolist()
    x_3 = fcn_lr_3['Step'].tolist()
    y_3 = fcn_lr_3['Value'].tolist()

    data = [(x_5, y_5), (x_4, y_4), (x_3, y_3)]
    legend = ['LR 1e-5', 'LR 1e-4', 'LR 1e-3']
    viz.plot(data=data,
             title='Effect of Learning Rate on Validation Loss',
             xlabel='steps',
             ylabel='loss',
             legend=legend,
             plot_name='fcn_validation_lr_effect',
             smoothing=20)

# ------------------------------------------------------------
# --- Resnet 8 Network size effect ---
if False:
    resnet_8 = pd.read_csv(
        "plot_data/experiments/resnet_network_size/2020-03-04--21-39-tag-training_loss_net_8.csv"
    )
    resnet_16 = pd.read_csv(
        "plot_data/experiments/resnet_network_size/2020-03-04--21-14-tag-training_loss_net_16.csv"
    )
    resnet_32 = pd.read_csv(
        "plot_data/experiments/resnet_network_size/2020-03-04--23-50-tag-training_loss_net_32.csv"
    )

    x_8 = resnet_8['Step'].tolist()[0:-1]
    y_8 = resnet_8['Value'].tolist()[0:-1]
    x_16 = resnet_16['Step'].tolist()[0:-1]
    y_16 = resnet_16['Value'].tolist()[0:-1]
    x_32 = resnet_32['Step'].tolist()[0:-1]
    y_32 = resnet_32['Value'].tolist()[0:-1]

    data = [(x_8, y_8), (x_16, y_16), (x_32, y_32)]
    legend = ['resnet-8', 'resnet-16', 'resnet-32']
    viz.plot(data=data,
             title='Resnet: Effect of Network Size on Training Loss',
             xlabel='steps',
             ylabel='loss',
             legend=legend,
             plot_name='resnet_network_size_effect',
             smoothing=5)

# --- resnet: effect of network size on validation loss ---
if False:
    resnet_8 = pd.read_csv(
        "plot_data/experiments/resnet_network_size/2020-03-04--21-39-tag-validation_loss_net_8.csv"
    )
    resnet_16 = pd.read_csv(
        "plot_data/experiments/resnet_network_size/2020-03-04--21-14-tag-validation_loss_net_16.csv"
    )
    resnet_32 = pd.read_csv(
        "plot_data/experiments/resnet_network_size/2020-03-04--23-50-tag-validation_loss_net_32.csv"
    )

    x_8 = resnet_8['Step'].tolist()[0:-1]
    y_8 = resnet_8['Value'].tolist()[0:-1]
    x_16 = resnet_16['Step'].tolist()[0:-1]
    y_16 = resnet_16['Value'].tolist()[0:-1]
    x_32 = resnet_32['Step'].tolist()[0:-1]
    y_32 = resnet_32['Value'].tolist()[0:-1]

    data = [(x_8, y_8), (x_16, y_16), (x_32, y_32)]
    legend = ['resnet-8', 'resnet-16', 'resnet-32']
    viz.plot(data=data,
             title='Resnet: Effect of Network Size on Training Loss',
             xlabel='steps',
             ylabel='loss',
             legend=legend,
             plot_name='resnet_network_size_effect',
             smoothing=5)


#  --- FCN Different skip values for Validation
if False:
    fcn_sk_10 = pd.read_csv(
        "plot_data/experiments/fcn_skip_effect/skip_frame_2020-03-07--21-53-tag-validation_loss.csv"
    )
    fcn_sk_5 = pd.read_csv(
        "plot_data/experiments/fcn_skip_effect/skip_frame_2020-03-07--22-03-tag-validation_loss.csv"
    )
    fcn_sk_2 = pd.read_csv(
        "plot_data/experiments/fcn_skip_effect/skip_frame_2020-03-07--22-43-tag-validation_loss.csv"
    )

    x_5 = fcn_sk_10['Step'].tolist()[0:680]
    y_5 = fcn_sk_10['Value'].tolist()[0:680]
    x_4 = fcn_sk_5['Step'].tolist()
    y_4 = fcn_sk_5['Value'].tolist()
    x_3 = fcn_sk_2['Step'].tolist()
    y_3 = fcn_sk_2['Value'].tolist()

    data = [(x_5, y_5), (x_4, y_4), (x_3, y_3)]
    legend = ['Skip 10', 'Skip 5', 'Skip 2']
    viz.plot(data=data,
             title='Effect of Skip frame HP on Validation Loss',
             xlabel='steps',
             ylabel='loss',
             legend=legend,
             plot_name='fcn_validation_sk_effect',
             smoothing=20)

#  --- FCN Different skip values for Training
if False:
    fcn_sk_10 = pd.read_csv(
        "plot_data/experiments/fcn_skip_effect/skip_frame_2020-03-07--21-53-tag-training_loss.csv"
    )
    fcn_sk_5 = pd.read_csv(
        "plot_data/experiments/fcn_skip_effect/skip_frame_2020-03-07--22-03-tag-training_loss.csv"
    )
    fcn_sk_2 = pd.read_csv(
        "plot_data/experiments/fcn_skip_effect/skip_frame_2020-03-07--22-43-tag-training_loss.csv"
    )

    x_5 = fcn_sk_10['Step'].tolist()[0:680]
    y_5 = fcn_sk_10['Value'].tolist()[0:680]
    x_4 = fcn_sk_5['Step'].tolist()
    y_4 = fcn_sk_5['Value'].tolist()
    x_3 = fcn_sk_2['Step'].tolist()
    y_3 = fcn_sk_2['Value'].tolist()

    data = [(x_5, y_5), (x_4, y_4), (x_3, y_3)]
    legend = ['skip 10', 'skip 5', 'skip 2']
    viz.plot(data=data,
             title='Effect of skip frame HP on Training Loss',
             xlabel='steps',
             ylabel='loss',
             legend=legend,
             plot_name='fcn_training_sk_effect',
             smoothing=20)


if True:
    legend = ['None', 'Left', 'Up', 'Right']
    viz.plot_actions_histogram(datasets_dir='./data',
                                title='Distribution of actions',
                                xlabel='actions',
                                ylabel='frequencies',
                                legend=legend,
                                plot_name='histogram_actions')
# --- Best Model in action (Resnet) ---
if False:
    resnet_train = pd.read_csv(
        "plot_data/experiments/best_resnet/2020-03-07--15-33-tag-training_loss_best_model.csv"
    )
    resnet_valid = pd.read_csv(
        "plot_data/experiments/best_resnet/2020-03-07--15-33-tag-validation_loss_best_model.csv"
    )

    x_train = resnet_train['Step'].tolist()
    y_train = resnet_train['Value'].tolist()
    x_valid = resnet_valid['Step'].tolist()
    y_valid = resnet_valid['Value'].tolist()

    data = [(x_train, y_train), (x_valid, y_valid)]
    legend = ['training', 'validation']
    viz.plot(data=data,
             title='Resnet: Best Model Based on Performance',
             xlabel='steps',
             ylabel='loss',
             legend=legend,
             plot_name='resnet_best_model',
             smoothing=10)

# --- Best Resnet Model: LR schedule ---
if False:
    resnet_lr = pd.read_csv(
        "plot_data/experiments/best_resnet/2020-03-07--15-33-tag-Learning_rate_schedule.csv"
    )

    x_lr = resnet_lr['Step'].tolist()
    y_lr = resnet_lr['Value'].tolist()

    data = [(x_lr, y_lr)]
    legend = ['lr schedule']
    viz.plot(data=data,
             title='Resnet: Learning Rate Schedule',
             xlabel='steps',
             ylabel='loss',
             legend=legend,
             plot_name='resnet_lr_schedule',
             smoothing=10)
