import numpy as np
from utils import *


class OptionalComAgent:
    """
    This class represents an agent that uses the SER optimisation criterion.
    """

    def __init__(self, no_com_agent, com_agent, id, u, du, alpha_q, alpha_msg, alpha_decay, num_objectives, opt=False):
        self.no_com_agent = no_com_agent
        self.com_agent = com_agent
        self.id = id
        self.u = u
        self.du = du
        self.num_messages = 2  # Only two types of communication: send a message or don't send a message
        self.num_objectives = num_objectives
        self.alpha_q = alpha_q
        self.alpha_msg = alpha_msg
        self.alpha_decay = alpha_decay
        self.theta = np.zeros(self.num_messages)
        self.policy = softmax(self.theta)
        self.op_policy = np.full(self.num_messages, 1.0 / self.num_messages)
        # optimistic initialization of Q-table
        if opt:
            self.q_table = np.ones((self.num_messages, num_objectives)) * 20
        else:
            self.q_table = np.zeros((self.num_messages, num_objectives))
        self.communicator = False

    def update(self, communicator, message, actions, reward):
        """
        This method will update the Q-table, strategy and internal parameters of the agent, as well as the secondary
        agent that played in this round.
        :param communicator: The id of the communicating agent.
        :param actions: The actions taken by the agents.
        :param reward: The reward obtained in this episode.
        :return: /
        """
        self.update_q_table(message, reward)
        theta, policy = self.update_policy(self.policy, self.theta, self.alpha_msg, self.q_table)
        self.theta = theta
        self.policy = policy
        if message is None:
            self.no_com_agent.update(communicator, message, actions, reward)
        else:
            self.com_agent.update(communicator, message, actions, reward)
        self.update_parameters()

    def update_q_table(self, message, reward):
        """
        This method will update the Q-table based on the message and the obtained reward.
        :param message: The message that was sent in the finished episode.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        if message is None:
            idx = 0
        else:
            idx = 1
        self.q_table[idx] += self.alpha_q * (reward - self.q_table[idx])

    def update_policy(self, policy, theta, alpha, expected_q):
        """
        This method will update the given theta parameters and policy.
        :policy: The policy we want to update
        :theta: The current parameters for this policy.
        :expected_q: The Q-values for this policy.
        :return: Updated theta parameters and policy.
        """
        policy = np.copy(policy)  # This avoids some weird numpy bugs where the policy/theta is referenced by pointer.
        theta = np.copy(theta)
        expected_u = policy @ expected_q
        # We apply the chain rule to calculate the gradient.
        grad_u = self.du(expected_u)  # The gradient of u
        grad_pg = softmax_grad(policy).T @ expected_q  # The gradient of the softmax function
        grad_theta = grad_u @ grad_pg.T  # The gradient of the complete function J(theta).
        theta += alpha * grad_theta
        policy = softmax(theta)
        return theta, policy

    def update_parameters(self):
        """
        This method will update the internal parameters of the agent.
        :return: /
        """
        self.alpha_q *= self.alpha_decay
        self.alpha_msg *= self.alpha_decay

    def get_message(self):
        """
        This method will choose what message is sent to the other agent.
        :return: Either None if opting to not communicate or the current policy if opting to communicate.
        """
        message = np.random.choice(range(self.num_messages), p=self.policy)
        if message == 0:  # Don't communicate
            return None
        else:
            self.communicator = True
            return self.com_agent.get_message()

    def select_action(self, message):
        """
        This method will select an action based on the message that was sent.
        :param message: The message that was sent.
        :return: The selected action.
        """
        if message is None:
            return self.no_com_agent.select_action(message)
        else:
            return self.com_agent.select_action(message)
