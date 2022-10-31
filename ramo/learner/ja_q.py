import numpy as np

from ramo.strategy.best_response import calc_best_response


class JointActionQAgent:
    """An independent learner using Q-learning for the SER criterion."""

    def __init__(self, id, u, num_actions, num_objectives, player_actions, alpha_q=0.01, alpha_q_decay=1, epsilon=0.01,
                 epsilon_decay=1, min_epsilon=0.01, rng=None):
        self.id = id
        self.u = u
        self.rng = rng if rng is not None else np.random.default_rng()
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.player_actions = player_actions
        self.num_players = len(player_actions)

        self.alpha_q = alpha_q
        self.alpha_q_decay = alpha_q_decay
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        ja_shape = player_actions + tuple([num_objectives])
        self.q_table = np.zeros(ja_shape)
        self.counts = np.zeros(player_actions)
        self.joint_policy = [np.full(actions, 1 / actions) for actions in player_actions]
        self.policy = np.full(num_actions, 1 / num_actions)

    def update(self, actions, reward):
        """Perform an update for the agent.

        Args:
            actions (List[int]): The actions that were taken by the agents.
            reward (float): The reward that was obtained by the agent.

        Returns:

        """
        self.update_q_table(actions, reward)
        self.update_counts(actions)
        self.update_policies()
        self.policy = calc_best_response(self.u, self.id, self.q_table, self.joint_policy, init_strat=self.policy)
        self.update_parameters()

    def update_q_table(self, actions, reward):
        """Update the joint-action Q-table.

        Args:
            actions (List[int]): The actions chosen by the agents.
            reward (float): The reward obtained by this agent.

        Returns:

        """
        idx = tuple(actions)
        self.q_table[idx] += self.alpha_q * (reward - self.q_table[idx])

    def update_counts(self, actions):
        """Update the joint-action counts.

        Args:
            actions (List[int]): The actions chosen by the agents.

        Returns:

        """
        idx = tuple(actions)
        self.counts[idx] += 1

    def update_policies(self):
        """Update the joint policies from the empirical action distribution."""
        joint_strategy = []
        total_count = np.sum(self.counts)

        for player in range(self.num_players):
            axis = tuple(np.delete(np.arange(self.num_players), player))
            action_counts = np.sum(self.counts, axis=axis)
            player_strat = action_counts / total_count
            joint_strategy.append(player_strat)

        self.joint_policy = joint_strategy

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
