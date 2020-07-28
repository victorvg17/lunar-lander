from __future__ import print_function
from collections import deque
import json
from datetime import datetime
import numpy as np
import torch

from lunar_lander import LunarLander
from agent.bc_agent import BCAgent
from config import Config
from utils import rgb2gray


class ImageBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.Q = deque()

    def push(self, data):
        if len(self.Q) == self.capacity:
            self.Q.popleft()
        self.Q.append(data)

    def is_full(self):
        return len(self.Q) == self.capacity

    def pop(self):
        result = np.concatenate(list(self.Q), axis=-1)
        return result


def run_episode(env, agent, config, rendering=True, max_timesteps=1000):

    episode_reward = 0
    step = 0
    is_fcn = config.is_fcn
    buffer = ImageBuffer(capacity=config.history_length + 1)

    state = env.reset()
    # downsampling (every 4th pixel). Copy because torch gives negative stride error
    state_img = env.render(mode="rgb_array")[::4, ::4, :].copy()

    # fix bug of corrupted states without rendering in gym environments
    env.viewer.window.dispatch_events()

    agent.test_mode()
    while True:
        if (is_fcn):
            a = agent.predict(X=np.expand_dims(state, axis=0))
        else:
            # preprocessing
            state_img = rgb2gray(state_img)
            state_img = np.expand_dims(a=state_img, axis=-1)
            state_img = np.expand_dims(a=state_img, axis=0)
            buffer.push(state_img)
            if buffer.is_full():
                state_img = buffer.pop()
                a = agent.predict(X=state_img)
            else:
                a = torch.zeros(4)
                a[0] = 1  # no-action aciton
        a = np.argmax(a.numpy())

        next_state, r, done, info = env.step(a)
        next_state_img = env.render(mode="rgb_array")[::4, ::4, :].copy()

        episode_reward += r
        state = next_state
        state_img = next_state_img
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":

    # important: probably it doesn't work for you to set rendering to False for evaluation
    rendering = True

    conf = Config()
    agent = BCAgent(conf)
    model_name = 'agent_2020-03-07--19-42.pt'
    agent.load(f"models/{model_name}", to_cpu=True)
    env = LunarLander()

    episode_rewards = []
    for i in range(conf.n_test_episodes):
        episode_reward = run_episode(env, agent, conf, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    timestamp = model_name.split(sep='_')[1][0:-3]
    fname = f"results/results_bc_agent-{timestamp}.json"
    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()
    env.close()
    print('... finished')
