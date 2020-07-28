import unittest
from agent.bc_agent import BCAgent
from agent.networks import FCN, CNN
from agent.resnet18 import Resnet18
from config import Config


class TestBCAgentMethods(unittest.TestCase):
    def test_fcn_agent_creation(self):
        conf = Config()
        conf.is_fcn = True
        agent = BCAgent(conf)
        self.assertIsInstance(agent.agent.net, FCN)

    def test_cnn_agent_creation(self):
        conf = Config()
        conf.is_fcn = False
        agent = BCAgent(conf)
        self.assertIsInstance(agent.agent.net, Resnet18)


if __name__ == '__main__':
    unittest.main()