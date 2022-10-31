import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.nn import softmax

from ramo.strategy.best_response import calc_expected_returns
from ramo.strategy.strategies import softmax_strategy


class JointActionActorCriticAgent:
    """A joint-action learner using the multi-objective actor-critic algorithm for the SER criterion."""

    def __init__(self, id, u, num_actions, num_objectives, player_actions, alpha_q=0.01, alpha_theta=0.01,
                 alpha_q_decay=1, alpha_theta_decay=1, rng=None):
        self.id = id
        self.u = u
        self.rng = rng if rng is not None else np.random.default_rng()
        self.grad = jit(grad(self.objective_function))
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.num_players = len(player_actions)

        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.alpha_q_decay = alpha_q_decay
        self.alpha_theta_decay = alpha_theta_decay

        ja_shape = player_actions + tuple([num_objectives])
        self.q_table = np.zeros(ja_shape)
        self.counts = np.zeros(player_actions)
        self.joint_policy = [np.full(actions, 1 / actions) for actions in player_actions]
        self.theta = np.zeros(num_actions)
        self.policy = softmax(self.theta)

    def objective_function(self, theta, q_values):
        """The objective function for the agent. This is the SER criterion.

        Args:
            theta (ndarray): The policy parameters.
            q_values (ndarray): The expected returns for the actions.

        Returns:
            float: The utility from the current policy and Q-values.

        """
        policy = softmax(theta)
        expected_returns = jnp.matmul(policy, q_values)
        utility = self.u(expected_returns)
        return utility

    def update(self, actions, reward):
        """Perform an update for the agent.

        Args:
            actions (List[int]): The actions taken by all players.
            reward (float): The reward that was obtained by the agent.

        Returns:

        """
        self.update_q_table(actions, reward)
        self.update_counts(actions)
        self.update_policies()
        q_values = calc_expected_returns(self.id, self.q_table, self.joint_policy)
        self.theta += self.alpha_theta * self.grad(self.q_table, q_values)
        self.policy = softmax_strategy(self.theta)
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
        """Update the hyperparameters."""
        self.alpha_q *= self.alpha_q_decay
        self.alpha_theta *= self.alpha_theta_decay

    def select_action(self):
        """Select an action according to the agent's policy.

        Returns:
            int: The selected action.

        """
        return self.rng.choice(range(self.num_actions), p=self.policy)
