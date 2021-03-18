import numpy as np
from utils import *


class CoopPolicyAgent:
    """
    This class represents an agent that uses the SER optimisation criterion.
    """

    def __init__(self, id, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives, opt=False):
        self.id = id
        self.u = u
        self.du = du
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.alpha_decay = alpha_decay
        self.theta = np.zeros(num_actions)
        self.policy = softmax(self.theta)
        self.best_response_theta = np.zeros(num_actions)
        self.best_response_policy = softmax(self.best_response_theta)
        self.op_policy = np.full(num_actions, 1 / num_actions)
        # optimistic initialization of Q-table
        if opt:
            self.q_table = np.ones((num_actions, num_actions, num_objectives)) * 20
        else:
            self.q_table = np.zeros((num_actions, num_actions, num_objectives))
        self.communicating = False

    def update(self, message, actions, reward):
        """
        This method will update the Q-table, strategy and internal parameters of the agent.
        :param message: The message that was sent. Unused by this agent.
        :param actions: The actions taken by the agents.
        :param reward: The reward obtained in this episode.
        :return: /
        """
        self.update_q_table(actions, reward)
        if self.id == 0:
            expected_q = self.op_policy @ self.q_table
        else:
            # We have to transpose axis 0 and 1 to interpret this as the column player.
            expected_q = self.op_policy @ self.q_table.transpose((1, 0, 2))
        theta, policy = self.update_policy(self.best_response_policy, self.best_response_theta, expected_q)
        self.theta = theta
        self.policy = policy
        self.best_response_theta = theta
        self.best_response_policy = policy
        self.update_parameters()

    def update_q_table(self, actions, reward):
        """
        This method will update the Q-table based on the message, chosen actions and the obtained reward.
        :param actions: The actions taken by the agents.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        self.q_table[actions[0], actions[1]] += self.alpha_q * (reward - self.q_table[actions[0], actions[1]])

    def update_policy(self, policy, theta, expected_q):
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

    def get_message(self):
        """
        This method will send the current policy of the agent as a message.
        :return: The current learned policy.
        """
        self.communicating = True
        return self.policy

    def select_action(self, message):
        """
        This method will select an action based on the message that was sent.
        :param message: The message that was sent.
        :return: The selected action.
        """
        if self.communicating:
            self.communicating = False
            return self.select_committed_action()
        else:
            return self.select_counter_action(message)

    def select_counter_action(self, op_policy):
        """
        This method will calculate a best-response policy to the received message and select an action from this policy.
        :param op_policy: The policy committed to by the opponent.
        :return: The selected action.
        """
        self.op_policy = op_policy
        if self.id == 0:
            expected_q = self.op_policy @ self.q_table
        else:
            # We have to transpose axis 0 and 1 to interpret this as the column player.
            expected_q = self.op_policy @ self.q_table.transpose((1, 0, 2))
        best_response_theta, best_response_policy = self.update_policy(self.policy, self.theta, expected_q)
        self.best_response_theta = best_response_theta
        self.best_response_policy = best_response_policy
        return np.random.choice(range(self.num_actions), p=self.best_response_policy)

    def select_committed_action(self):
        """
        This method uses the committed policy to select the action that will be played.
        :return: An action that was selected using the current policy.
        """
        return np.random.choice(range(self.num_actions), p=self.policy)
