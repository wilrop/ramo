from collections import deque

import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.nn import softmax

from ramo.strategy.strategies import softmax_strategy
from ramo.utils.helpers import array_slice


class NonStationaryAgent:
    """An agent that learns a non-stationary policy to each pure-strategy commitment from the leader."""

    def __init__(self, id, u, num_actions, num_objectives, alpha_q=0.01, alpha_theta=0.01, alpha_q_decay=1,
                 alpha_theta_decay=1, buffer_size=20, rng=None):
        self.id = id
        self.u = u
        self.rng = rng if rng is not None else np.random.default_rng()
        self.grad_leader = jit(grad(self.objective_function_leader))
        self.grad_follower = jit(grad(self.objective_function_follower))
        self.num_actions = num_actions
        self.num_opponent_actions = num_actions  # Defaults to same number of actions.
        self.num_objectives = num_objectives

        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.alpha_q_decay = alpha_q_decay
        self.alpha_theta_decay = alpha_theta_decay

        self.payoffs_table = np.zeros((num_actions, num_actions, num_objectives))
        self.leader_q_table = np.zeros((num_actions, num_objectives))
        self.leader_theta = np.zeros(num_actions)
        self.leader_policy = np.full(num_actions, 1 / num_actions)
        self.follower_thetas = np.zeros((num_actions, num_actions))
        self.follower_policies = np.full((num_actions, num_actions), 1 / num_actions)
        self.empirical_opponent_policy = deque(maxlen=buffer_size)

        self.leader = False

    def objective_function_leader(self, theta, q_values):
        """The objective function for the leader.

        Args:
            theta (ndarray): The parameters for the commitment policy.
            q_values (ndarray): The Q-values for committing to actions.

        Returns:
            float: The utility from the commitment strategy.

        """
        policy = softmax(theta)
        expected_returns = jnp.matmul(policy, q_values)
        utility = self.u(expected_returns)
        return utility

    def objective_function_follower(self, thetas, q_values, leader_policy):
        """The objective function for the follower.

        Args:
            thetas (ndarray): A matrix of thetas.
            q_values (ndarray): Learned Q-values for the joint-actions.
            leader_policy (ndarray): The committed non-stationary strategy from the leader.

        Returns:
            float: The utility from these parameters.

        """
        expected_returns = jnp.zeros(self.num_objectives)
        for i in range(self.num_opponent_actions):
            expected_q = q_values[i]
            prob = leader_policy[i]
            theta = thetas[i]
            policy = softmax(theta)
            expected_returns = expected_returns + prob * jnp.matmul(policy, expected_q)
        utility = self.u(expected_returns)
        return utility

    def make_leader(self):
        """Make this agent the leader."""
        self.leader = True

    def make_follower(self):
        """Make this agent the follower."""
        self.leader = False

    def set_opponent_actions(self, num_actions):
        """Set the number of actions that the opponent can play.

        Args:
            num_actions (int): The number of actions for the opponent.

        Returns:

        """
        self.num_opponent_actions = num_actions

    def update(self, commitment, actions, reward):
        """Perform an update of the agent. Specifically updates the Q-tables, policies and hyperparameters.

        Args:
            commitment (int): The leader's non-stationary commitment strategy.
            actions (List[int]): The actions selected in an episode.
            reward (float): The reward that was obtained by the agent in that episode.

        Returns:

        """
        own_action = actions[self.id]
        self.update_payoffs_table(actions, reward)

        if self.leader:
            self.update_leader_q_table(own_action, reward)
            self.leader_theta += self.alpha_theta * self.grad_leader(self.leader_theta, self.leader_q_table)
            self.leader_policy = softmax_strategy(self.leader_theta)
        else:
            # Get the correct view of the payoffs table for this player.
            q_values = array_slice(self.payoffs_table, abs(1 - self.id), 0, self.num_opponent_actions)

            # Calculate the empirical opponent policy.
            self.empirical_opponent_policy.append(commitment)
            opp_policy = np.bincount(self.empirical_opponent_policy, minlength=self.num_opponent_actions)

            # Update the follower policies that make up their non-stationary policy.
            self.follower_thetas += self.alpha_theta * self.grad_follower(self.follower_thetas, q_values, opp_policy)
            for idx, theta in enumerate(self.follower_thetas):
                self.follower_policies[idx] = softmax_strategy(theta)

        self.update_parameters()

    def update_leader_q_table(self, action, reward):
        """Update the leader's Q-table based on their own action and the obtained reward.

        Args:
            action (int): The action taken by the leader.
            reward (float): The reward obtained by this agent.

        Returns:

        """
        self.leader_q_table[action] += self.alpha_q * (reward - self.leader_q_table[action])

    def update_payoffs_table(self, actions, reward):
        """Update the joint-action payoffs table.

        Args:
            actions (List[int]): The actions that were taken in an episode.
            reward (float): The reward obtained by this joint action.

        Returns:

        """
        idx = tuple(actions)
        self.payoffs_table[idx] += self.alpha_q * (reward - self.payoffs_table[idx])

    def update_parameters(self):
        """Update the internal parameters of the agent."""
        self.alpha_q *= self.alpha_q_decay
        self.alpha_theta *= self.alpha_theta_decay

    def get_commitment(self):
        """Get the commitment from the leader.

        Returns:
            int: A pure strategy commitment of the leader.

        """
        return self.rng.choice(range(self.num_actions), p=self.leader_policy)

    def select_action(self, commitment):
        """Select an action based on the commitment of the leader.

        Args:
            commitment (int): The message that was sent.

        Returns:
            int: The selected action.

        """
        if self.leader:
            return self.select_committed(commitment)  # If this agent is committing, they must follow through.
        else:
            return self.select_counter_action(commitment)  # Otherwise select a counter action.

    def select_counter_action(self, leader_action):
        """Select the correct counter policy and sample an action using this policy.

        Args:
            leader_action (int): The committed pure strategy from the leader.

        Returns:
            int: The selected action.

        """
        policy = self.follower_policies[leader_action]
        return self.rng.choice(range(self.num_actions), p=policy)

    def select_committed(self, leader_action):
        """Play the pure strategy that was committed.

        Args:
            leader_action (int): The pure strategy (action) the leader published.

        Returns:
            int: The committed action.

        """
        return leader_action
