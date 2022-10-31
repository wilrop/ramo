import numpy as np

from ramo.strategy.best_response import optimise_policy


class IndependentQAgent:
    """An independent learner using Q-learning for the SER criterion.

    This implementation is based on the multi-objective Q-learning algorithm proposed in [1].

    References:
        .. [1] Rădulescu, R., Mannion, P., Zhang, Y., Roijers, D., & Nowé, A. (2020). A utility-based analysis of
            equilibria in multi-objective normal-form games. The Knowledge Engineering Review, 35, e32.

    """

    def __init__(self, u, num_actions, num_objectives, alpha_q=0.01, alpha_q_decay=1, epsilon=0.01, epsilon_decay=1,
                 min_epsilon=0.01, rng=None):
        self.u = u
        self.rng = rng if rng is not None else np.random.default_rng()
        self.num_actions = num_actions
        self.num_objectives = num_objectives

        self.alpha_q = alpha_q
        self.alpha_q_decay = alpha_q_decay
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_table = np.zeros((num_actions, num_objectives))
        self.policy = np.full(num_actions, 1 / num_actions)

    def update(self, action, reward):
        """Perform an update for the agent.

        Args:
            action (int): The actions that was taken by the agent.
            reward (float): The reward that was obtained by the agent.

        Returns:

        """
        self.update_q_table(action, reward)
        self.policy = optimise_policy(self.q_table, self.u, init_strat=self.policy)
        self.update_parameters()

    def update_q_table(self, action, reward):
        """Update the Q-table based on the chosen actions and the obtained reward.

        Args:
            action (int): The action chosen by this agent.
            reward (float): The reward obtained by this agent.

        Returns:

        """
        self.q_table[action] += self.alpha_q * (reward - self.q_table[action])

    def update_parameters(self):
        """Update the hyperparameters. This decays the learning rate for the Q-values and exploration parameter."""
        self.alpha_q *= self.alpha_q_decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def select_action(self):
        """Select an action according to the agent's policy.

        Returns:
            int: The selected action.

        """
        if self.rng.uniform(0, 1) < self.epsilon:
            return self.rng.integers(self.num_actions)
        else:
            return self.rng.choice(range(self.num_actions), p=self.policy)
