import numpy as np
from utils import *


class NoComAgent:
    """
    This class represents an agent that uses the SER multi-objective optimisation criterion.
    """

    def __init__(self, id, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives, opt=False):
        self.id = id
        self.u = u
        self.du = du
        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.alpha_decay = alpha_decay
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.theta = np.zeros(num_actions)
        self.policy = softmax(self.theta)
        # optimistic initialization of Q-table
        if opt:
            self.q_table = np.ones((num_actions, num_objectives)) * 20
        else:
            self.q_table = np.zeros((num_actions, num_objectives))

    def update(self, message, actions, reward):
        """
        This method will update the Q-table, strategy and internal parameters of the agent.
        :param message: The message that was sent. Unused by this agent.
        :param actions: The actions that were executed.
        :param reward: The reward that was obtained by the agent.
        :return: /
        """
        action = actions[self.id]
        self.update_q_table(action, reward)
        self.update_policy()
        self.update_parameters()

    def update_q_table(self, action, reward):
        """
        This method will update the Q-table based on the chosen actions and the obtained reward.
        :param action: The action chosen by this agent.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        self.q_table[action] += self.alpha_q * (reward - self.q_table[action])

    def update_policy(self):
        """
        This method will update the given theta parameters and policy.
        :return: /
        """
        expected_u = self.policy @ self.q_table
        # We apply the chain rule to calculate the gradient.
        grad_u = self.du(expected_u)  # The gradient of u
        grad_pg = softmax_grad(self.policy).T @ self.q_table  # The gradient of the softmax function
        grad_theta = grad_u @ grad_pg.T  # The gradient of the complete function J(theta).
        self.theta += self.alpha_theta * grad_theta
        self.policy = softmax(self.theta)

    def update_parameters(self):
        """
        This method will update the internal parameters of the agent.
        :return: /
        """
        self.alpha_q *= self.alpha_decay
        self.alpha_theta *= self.alpha_decay

    @staticmethod
    def get_message():
        """
        This method will get a message from this agent.
        :return: This agent doesn't communicate so it always returns None.
        """
        return None

    def select_action(self, message):
        """
        This method will select an action according to the agent's policy.
        :param message: The communication from this episode (unused by this agent).
        :return: The selected action.
        """
        return np.random.choice(range(self.num_actions), p=self.policy)
