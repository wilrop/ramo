import numpy as np
from utils import *


class CompActionAgent:
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
        # optimistic initialization of Q-table
        if opt:
            self.msg_q_table = np.ones((num_actions, num_objectives)) * 20
        else:
            self.msg_q_table = np.zeros((num_actions, num_objectives))
        self.payoffs_table = np.zeros((num_actions, num_actions, num_objectives))
        self.msg_theta = np.zeros(num_actions)
        self.msg_policy = np.full(num_actions, 1 / num_actions)
        self.counter_thetas = np.zeros((num_actions, num_actions))
        self.counter_policies = np.full((num_actions, num_actions), 1 / num_actions)
        self.communicating = False

    def update(self, communicator, message, actions, reward):
        """
        This method will update the Q-table, strategy and internal parameters of the agent.
        :param communicator: The id of the communicating agent.
        :param message: The message that was sent.
        :param actions: The actions selected in the previous episode.
        :param reward: The reward that was obtained by the agent.
        :return: /
        """
        self.update_payoffs_table(actions, reward)
        own_action = actions[self.id]
        if communicator == self.id:
            self.update_msg_q_table(own_action, reward)
            theta, policy = self.update_policy(self.msg_policy, self.msg_theta, self.msg_q_table)
            self.msg_theta = theta
            self.msg_policy = policy
        else:
            policy = self.counter_policies[message]
            theta = self.counter_thetas[message]
            if self.id == 0:
                expected_q = self.payoffs_table.transpose((1, 0, 2))[message]
            else:
                expected_q = self.payoffs_table[message]
            theta, policy = self.update_policy(policy, theta, expected_q)
            self.counter_thetas[message] = theta
            self.counter_policies[message] = policy

        self.update_parameters()

    def update_msg_q_table(self, action, reward):
        """
        This method will update the Q-table based on the chosen actions and the obtained reward.
        :param action: The action chosen by this agent.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        self.msg_q_table[action] += self.alpha_q * (reward - self.msg_q_table[action])

    def update_payoffs_table(self, actions, reward):
        """
        This method will update the payoffs table to learn the payoff vector of joint actions.
        :param actions: The actions that were taken in the previous episode.
        :param reward: The reward obtained by this joint action.
        :return: /
        """
        self.payoffs_table[actions[0], actions[1]] += self.alpha_q * (
                reward - self.payoffs_table[actions[0], actions[1]])

    def update_policy(self, policy, theta, expected_q):
        """
        This method will update the given theta parameters and policy.
        :param policy: The policy we want to update.
        :param theta: The current theta parameters for this policy.
        :param expected_q: The Q-values for this policy.
        :return: Updated theta parameters and policy.
        """
        policy = np.copy(policy)  # This avoids some weird numpy bugs where the policy/theta is referenced by pointer.
        theta = np.copy(theta)
        expected_u = policy @ expected_q
        # We apply the chain rule to calculate the gradient.
        grad_u = self.du(expected_u)  # The gradient of u
        grad_pg = softmax_grad(policy).T @ expected_q  # The gradient of the softmax function
        grad_theta = grad_u @ grad_pg.T  # The gradient of the complete function J(theta).
        theta += self.alpha_theta * grad_theta
        policy = softmax(theta)
        return theta, policy

    def update_parameters(self):
        """
        This method will update the internal parameters of the agent.
        :return: /
        """
        self.alpha_q *= self.alpha_decay
        self.alpha_theta *= self.alpha_decay

    def select_action(self, message):
        """
        This method will select an action based on the message that was sent.
        :param message: The message that was sent.
        :return: The selected action.
        """
        if self.communicating:
            self.communicating = False
            return self.select_published_action(message)  # If this agent is committing, they must follow through.
        else:
            return self.select_counter_action(message)  # Otherwise select a counter action.

    def get_message(self):
        """
        This method will determine what action this agent will publish.
        :return: The action that will maximise this agent's SER, given that the other agent also maximises its response.
        """
        self.communicating = True
        return np.random.choice(range(self.num_actions), p=self.msg_policy)

    def select_counter_action(self, state):
        """
        This method will select the best counter policy and choose an action using this policy.
        :param state: The message from an agent in the form of their next action.
        :return: The selected action.
        """
        policy = self.counter_policies[state]
        return np.random.choice(range(self.num_actions), p=policy)

    @staticmethod
    def select_published_action(state):
        """
        This method simply plays the action that it already published.
        :param state: The action it published.
        :return: The action it published.
        """
        return state
