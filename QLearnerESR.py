import random
import numpy as np
from scipy.optimize import minimize
import warnings
#from utils import *
import itertools


class QLearnerESR:
    def __init__(self, agent_id, alpha, gamma, epsilon, num_states, num_actions, num_objectives, opt=False,
                 rand_prob=False):
        self.agent_id = agent_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.joint_table = np.zeros((num_actions, num_actions))
        # optimistic initialization of Q-table
        if opt:
            self.q_table = np.ones((num_states, num_actions)) * 20
        else:
            self.q_table = np.zeros((num_states, num_actions))
        self.current_state = -1
        self.rand_prob = rand_prob

    def update_q_table(self, prev_state, action, reward):
        """
        This method will update the Q-table based on the message, chosen actions and the obtained reward.
        :param prev_state: The message.
        :param action: The chosen action by this agent.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        old_q = self.q_table[prev_state][action]
        utility = calc_utility(self.agent_id, reward)
        new_q = old_q + self.alpha * (utility - old_q)
        self.q_table[prev_state][action] = new_q

    def update_joint_table(self, actions, payoffs):
        coords = tuple(actions)
        utility = calc_utility(self.agent_id, payoffs)
        self.joint_table[coords] = utility

    def pref_joint_action(self):
        """
        This method will calculate the preferred joint-action for an agent based on the payoffs of joint actions.
        :return: The joint action that will result in the highest utility for the agent.
        """
        return np.argmax(self.joint_table)

    def select_random_action(self):
        """
        This method will return a random action.
        :return: An action (an integer value).
        """
        random_action = np.random.randint(self.num_actions)
        return random_action

    def select_action_mixed_nonlinear(self, state):
        """
        This method will perform epsilon greedy search based on nonlinear optimiser mixed strategy.
        :param state: The message from an agent in the form of their preferred joint action.
        :return: The selected action.
        """
        self.current_state = state
        if random.uniform(0.0, 1.0) < self.epsilon:
            return self.select_random_action()
        else:
            return self.select_action_greedy_mixed_nonlinear(state)

    def select_action_greedy_mixed_nonlinear(self, state):
        """
        This method will perform greedy action selection based on nonlinear optimiser mixed strategy search.
        :param state: The preferred joint action.
        :return: The selected action.
        """
        strategy = self.calc_mixed_strategy_nonlinear()
        if isinstance(strategy, int) or isinstance(strategy, np.int64):
            return strategy
        else:
            if np.sum(strategy) != 1:
                strategy = strategy / np.sum(strategy)
            return np.random.choice(range(self.num_actions), p=strategy)

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
            s0 = np.full(self.num_actions, 1.0/self.num_actions)  # initial guess set to equal prob over all actions

        b = (0.0, 1.0)
        bnds = (b,) * self.num_actions  # Each pair in x will have this b as min, max
        con1 = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        cons = ([con1])
        solution = minimize(self.objective, s0, bounds=bnds, constraints=cons)
        strategy = solution.x

        return strategy

    def objective(self, strategy):
        """
        This method is the objective function to be minimised by the nonlinear optimiser.
        Therefore it returns the negative of SER.
        :param strategy: The mixed strategy for the agent.
        :return: The SER.
        """
        return - self.calc_esr_from_strategy(strategy)

    # Calculates the SER for a given strategy using the agent's own Q values
    def calc_esr_from_strategy(self, strategy):
        """
        This method will calculate the SER from a mixed strategy.
        :param strategy: The mixed strategy.
        :return: The SER.
        """
        esr = self.calc_esr(self.current_state, strategy)
        return esr

    # Calculates the expected payoff vector for a given strategy using the agent's own Q values
    def calc_esr(self, state, strategy):
        """
        This method calculates the expected scalarised reward for a given strategy using the agent's own Q values.
        :param state: The preferred joint action.
        :param strategy: The mixed strategy.
        :return: The expected scalarised reward for the strategy for all objectives.
        """
        esr = np.dot(self.q_table[state], np.array(strategy))
        return esr

    def select_preferred_action(self, state):
        joint_action = np.unravel_index(state, self.q_table.shape)  # Unravel the flat index to a coordinate array
        preferred_action = joint_action[self.agent_id]
        return preferred_action


def calc_utility(agent, vector):
    """
    This function will calculate the SER for an agent and their expected results vector.
    :param agent: The agent id.
    :param vector: Their expected results for the objectives.
    :return: The SER.
    """
    utility = 0
    if agent == 0:
        utility = vector[0] ** 2 + vector[1] ** 2  # Utility function for agent 1
    elif agent == 1:
        utility = vector[0] * vector[1]  # Utility function for agent 2
    return utility


def softmax(q):
    soft_q = np.exp(q - np.max(q))
    return soft_q / soft_q.sum(axis=0)
