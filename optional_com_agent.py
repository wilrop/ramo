import random
import numpy as np
from utils import *
from scipy.optimize import minimize


class ActorCriticSER:
    """
    This class represents an agent that uses the SER optimisation criterion.
    """

    def __init__(self, id, utility_function, derivative, alpha_msg, alpha_q, alpha_theta, num_actions, num_objectives, opt=False):
        self.id = id
        self.utility_function = utility_function
        self.derivative = derivative
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.alpha_msg = alpha_msg
        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.theta_m = np.zeros(num_actions)
        self.theta_nm = np.zeros(num_actions)
        self.policy_m = softmax(self.theta_m)
        self.policy_nm = softmax(self.theta_nm)
        self.num_messages = 2
        self.theta_msg = np.zeros(self.num_messages)
        self.policy_msg = softmax(self.theta_msg)
        self.op_policy = np.full(num_actions, 1.0 / num_actions)
        # optimistic initialization of Q-table
        if opt:
            self.msg_q_table = np.ones((self.num_messages, num_objectives)) * 20
            self.q_table_m = np.ones((num_actions, num_actions, num_objectives)) * 20
            self.q_table_nm = np.ones((num_actions, num_objectives)) * 20
        else:
            self.msg_q_table = np.zeros((self.num_messages, num_objectives))
            self.q_table_m = np.zeros((num_actions, num_actions, num_objectives))
            self.q_table_nm = np.zeros((num_actions, num_objectives))
        self.communicator = False

    def update(self, message, actions, reward):
        """
        This method updates the Q table and policy of the agent.
        :param actions: The actions taken by the agents.
        :param reward: The reward obtained in this episode.
        :return: /
        """
        self.update_msg_q_table(message, reward)
        msg_theta, msg_policy = self.update_policy(self.policy_msg, self.theta_msg, self.alpha_msg, self.msg_q_table)
        self.theta_msg = msg_theta
        self.policy_msg = msg_policy
        if message is None:
            self.update_q_table(message, self.q_table_nm, actions, reward)
            theta_nm, policy_nm = self.update_policy(self.policy_nm, self.theta_nm, self.alpha_theta, self.q_table_nm)
            self.theta_nm = theta_nm
            self.policy_nm = policy_nm
        else:
            self.update_q_table(message, self.q_table_m, actions, reward)
            if self.id == 0:
                expected_q = self.op_policy @ self.q_table_m
            else:
                # We have to transpose axis 0 and 1 to interpret this as the column player.
                expected_q = self.op_policy @ self.q_table_m.transpose((1, 0, 2))
            theta_m, policy_m = self.update_policy(self.policy_m, self.theta_m, self.alpha_theta, expected_q)
            self.theta_m = theta_m
            self.policy_m = policy_m
        # self.policy = self.msg_strategy[0] * self.policy_nm + self.msg_strategy[1] * self.policy_m

    def update_msg_q_table(self, message, reward):
        """
        This method will update the Q-table based on the message, chosen actions and the obtained reward.
        :param message: The message that was sent in the finished episode.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        if message is None:
            idx = 0
        else:
            idx = 1
        self.msg_q_table[idx] += self.alpha_q * (reward - self.msg_q_table[idx])

    def update_q_table(self, message, q_table, actions, reward):
        """
        This method will update the Q-table based on the message, chosen actions and the obtained reward.
        :param actions: The actions taken by the agents.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        if message is None:
            q_table[actions[self.id]] += self.alpha_q * (reward - q_table[actions[self.id]])
        else:
            q_table[actions[0], actions[1]] += self.alpha_q * (reward - q_table[actions[0], actions[1]])

    def update_policy(self, policy, theta, alpha, expected_q):
        """
        This method will update the parameters theta of the policy.
        :return: /
        """
        expected_u = policy @ expected_q
        # We apply the chain rule to calculate the gradient.
        grad_u = self.derivative(expected_u)  # The gradient of u
        grad_pg = softmax_grad(policy).T @ expected_q  # The gradient of the softmax function
        grad_theta = grad_u @ grad_pg.T  # The gradient of the complete function J(theta).
        theta += alpha * grad_theta
        policy = softmax(theta)
        return theta, policy

    def communicate(self):
        message = np.random.choice(range(self.num_messages), p=self.policy_msg)
        if message == 0:  # Don't communicate
            return None
        else:
            self.communicator = True
            return self.policy_m

    def select_action(self, message):
        """
        This method will select an action based on the message that was sent.
        :param message: The message that was sent.
        :return: The selected action.
        """
        if message is None:
            return np.random.choice(range(self.num_actions), p=self.policy_nm)
        else:
            if self.communicator:
                self.communicator = False
                return self.select_committed_strategy(message)
            else:
                return self.select_counter_action(message)

    def select_counter_action(self, op_policy):
        """
        This method will perform epsilon greedy action selection.
        :param op_policy: The strategy committed to by the opponent.
        :return: The selected action.
        """
        self.op_policy = op_policy
        if self.id == 0:
            expected_q = self.op_policy @ self.q_table_m
        else:
            # We have to transpose axis 0 and 1 to interpret this as the column player.
            expected_q = self.op_policy @ self.q_table_m.transpose((1, 0, 2))
        theta, policy = self.update_policy(self.policy_m, self.theta_m, self.alpha_theta, expected_q)
        self.theta_m = theta
        self.policy_m = policy
        return np.random.choice(range(self.num_actions), p=self.policy_m)

    def select_committed_strategy(self, strategy):
        """
        This method uses the committed strategy to select the action that will be played.
        :return: An action that was selected using the current policy.
        """
        return np.random.choice(range(self.num_actions), p=strategy)