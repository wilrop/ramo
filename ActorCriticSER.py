import random
import numpy as np
from utils import *


class ActorCriticSER:
    """
    This class represents an agent that uses the SER multi-objective optimisation criterion.
    """

    def __init__(self, id, utility_function, derivative, alpha_q, alpha_theta, num_actions, num_objectives, opt=False):
        self.id = id
        self.utility_function = utility_function
        self.derivative = derivative
        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.theta = np.zeros(num_actions)
        self.policy = softmax(self.theta)
        # optimistic initialization of Q-table
        if opt:
            self.q_table = np.ones((num_actions, num_objectives)) * 20
        else:
            self.q_table = np.zeros((num_actions, num_objectives))

    def update(self, action, reward):
        """
        This method will update the Q-table and strategy of the agent.
        :param action: The action that was chosen by the agent.
        :param reward: The reward that was obtained by the agent.
        :return: /
        """
        self.update_q_table(action, reward)
        self.update_policy()

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
        This method will update the internal policy of the agent.
        :return: /
        """
        expected_u = self.policy @ self.q_table
        # We apply the chain rule to calculate the gradient.
        grad_u = self.derivative(expected_u)  # The gradient of u
        grad_pg = softmax_grad(self.policy).T @ self.q_table  # The gradient of the softmax function
        grad_theta = grad_u @ grad_pg.T  # The gradient of the complete function J(theta).
        self.theta += self.alpha_theta * grad_theta
        self.policy = softmax(self.theta)

    def select_action(self):
        """
        This method will select an action according to the agent's policy.
        :return: The selected action.
        """
        return np.random.choice(range(self.num_actions), p=self.policy)
