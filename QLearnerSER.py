import random
import numpy as np
from scipy.optimize import minimize


class QLearnerSER:
    """
    This class represents an agent that uses the SER optimisation criterion.
    """

    def __init__(self, utility, alpha, epsilon, num_actions, num_objectives, opt=False, rand_prob=False):
        self.utility = utility
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.joint_table = np.zeros((num_actions, num_actions))
        # optimistic initialization of Q-table
        if opt:
            self.q_table = np.ones((num_actions, num_objectives)) * 20
        else:
            self.q_table = np.zeros((num_actions, num_objectives))
        self.rand_prob = rand_prob
        self.strategy = np.full(num_actions, 1 / num_actions)

    def update(self, action_dist, reward):
        self.update_q_table(action_dist, reward)
        self.strategy = self.calc_mixed_strategy_nonlinear()

    def update_q_table(self, action_dist, reward):
        """
        This method will update the Q-table based on the message, chosen actions and the obtained reward.
        :param action: The chosen action by this agent.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        for idx in range(self.num_actions):
            self.q_table[idx] += self.alpha * (reward * action_dist[idx] - self.q_table[idx])

    def select_random_action(self):
        """
        This method will return a random action.
        :return: An action (an integer value).
        """
        random_action = np.random.randint(self.num_actions)
        return random_action

    def select_action_mixed_nonlinear(self):
        """
        This method will perform epsilon greedy search based on nonlinear optimiser mixed strategy.
        :param state: The message from an agent in the form of their preferred joint action.
        :return: The selected action.
        """
        if random.uniform(0.0, 1.0) < self.epsilon:
            return self.select_random_action()
        else:
            return self.select_action_greedy_mixed_nonlinear()

    def select_action_greedy_mixed_nonlinear(self):
        """
        This method will perform greedy action selection based on nonlinear optimiser mixed strategy search.
        :param state: The preferred joint action.
        :return: The selected action.
        """
        return np.random.choice(range(self.num_actions), p=self.strategy)

    def calc_mixed_strategy_nonlinear(self):
        """
        This method will calculate a mixed strategy based on the nonlinear optimization.
        :param state: The preferred joint action.
        :return: A mixed strategy.
        """
        if self.rand_prob:
            s0 = np.random.random(self.num_actions)
            s0 /= np.sum(s0)
        else:
            s0 = np.full(self.num_actions, 1.0 / self.num_actions)  # initial guess set to equal prob over all actions

        b = (0.0, 1.0)
        bnds = (b,) * self.num_actions  # Each pair in x will have this b as min, max
        con1 = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        cons = ([con1])
        solution = minimize(self.objective, s0, bounds=bnds, constraints=cons)
        strategy = solution.x

        if np.sum(strategy) != 1:
            strategy = strategy / np.sum(strategy)
        return strategy

    def objective(self, strategy):
        """
        This method is the objective function to be minimised by the nonlinear optimiser.
        Therefore it returns the negative of SER.
        :param strategy: The mixed strategy for the agent.
        :return: The SER.
        """
        return - self.calc_ser_from_strategy(strategy)

    # Calculates the SER for a given strategy using the agent's own Q values
    def calc_ser_from_strategy(self, strategy):
        """
        This method will calculate the SER from a mixed strategy.
        :param strategy: The mixed strategy.
        :return: The SER.
        """
        expected_vec = self.calc_expected_vec(strategy)
        ser = self.utility(expected_vec)
        return ser

    # Calculates the expected payoff vector for a given strategy using the agent's own Q values
    def calc_expected_vec(self, strategy):
        """
        This method calculates the expected payoff vector for a given strategy using the agent's own Q values.
        :param state: The preferred joint action.
        :param strategy: The mixed strategy.
        :return: The expected results for all objectives.
        """
        expected_vec = strategy @ self.q_table
        return expected_vec
