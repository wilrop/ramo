import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.nn import softmax

from ramo.strategy.strategies import softmax_strategy
from ramo.utils.helpers import array_slice


class CompActionAgent:
    """An agent that learns a best-response policy to each pure-strategy commitment from the leader [1].

    References:
        .. [1] Willem Röpke, Diederik M. Roijers, Ann Nowé, & Roxana Rădulescu. (2021). Preference Communication in
            Multi-Objective Normal-Form Games.

    """

    def __init__(self, id, u, num_actions, num_objectives, alpha_lq=0.01, alpha_ltheta=0.01, alpha_fq=0.01,
                 alpha_ftheta=0.01, alpha_q_decay=1, alpha_theta_decay=1, rng=None):
        self.id = id
        self.u = u
        self.rng = rng if rng is not None else np.random.default_rng()
        self.grad = jit(grad(self.objective_function))
        self.num_actions = num_actions
        self.num_objectives = num_objectives

        self.alpha_lq = alpha_lq
        self.alpha_ltheta = alpha_ltheta
        self.alpha_fq = alpha_fq
        self.alpha_ftheta = alpha_ftheta
        self.alpha_q_decay = alpha_q_decay
        self.alpha_theta_decay = alpha_theta_decay

        self.payoffs_table = np.zeros((num_actions, num_actions, num_objectives))
        self.leader_q_table = np.zeros((num_actions, num_objectives))
        self.leader_theta = np.zeros(num_actions)
        self.leader_policy = np.full(num_actions, 1 / num_actions)
        self.follower_thetas = np.zeros((num_actions, num_actions))
        self.follower_policies = np.full((num_actions, num_actions), 1 / num_actions)

        self.leader = False

    def objective_function(self, theta, q_values):
        """The objective function.

        Args:
            theta (ndarray): The policy parameters.
            q_values (ndarray): Learned Q-values used to calculate the SER from these parameters.

        Returns:
            float: The utility from the current parameters theta and Q-values.

        """
        policy = softmax(theta)
        expected_returns = jnp.matmul(policy, q_values)
        utility = self.u(expected_returns)
        return utility

    def make_leader(self):
        """Make this agent the leader."""
        self.leader = True

    def make_follower(self):
        """Make this agent the follower."""
        self.leader = False

    def update(self, commitment, actions, reward):
        """Perform an update of the agent. Specifically updates the Q-tables, policies and hyperparameters.

        Args:
            commitment (int): The leader's committed action.
            actions (List[int]): The actions selected in an episode.
            reward (float): The reward that was obtained by the agent in that episode.

        Returns:

        """
        own_action = actions[self.id]
        if self.leader:
            self.update_leader_q_table(own_action, reward)
            self.leader_theta += self.alpha_ltheta * self.grad(self.leader_theta, self.leader_q_table)
            self.leader_policy = softmax_strategy(self.leader_theta)
        else:
            self.update_payoffs_table(actions, reward)
            q = array_slice(self.payoffs_table, abs(1 - self.id), commitment, commitment + 1)
            q = q.reshape(self.num_actions, self.num_objectives)
            self.follower_thetas[commitment] += self.alpha_ftheta * self.grad(self.follower_thetas[commitment], q)
            self.follower_policies[commitment] = softmax_strategy(self.follower_thetas[commitment])
        self.update_parameters()

    def update_leader_q_table(self, action, reward):
        """Update the leader's Q-table based on their own action and the obtained reward.

        Args:
            action (int): The action taken by the leader.
            reward (float): The reward obtained by this agent.

        Returns:

        """
        self.leader_q_table[action] += self.alpha_lq * (reward - self.leader_q_table[action])

    def update_payoffs_table(self, actions, reward):
        """Update the joint-action payoffs table.

        Args:
            actions (List[int]): The actions that were taken in an episode.
            reward (float): The reward obtained by this joint action.

        Returns:

        """
        idx = tuple(actions)
        self.payoffs_table[idx] += self.alpha_fq * (reward - self.payoffs_table[idx])

    def update_parameters(self):
        """Update the internal parameters of the agent."""
        self.alpha_lq *= self.alpha_q_decay
        self.alpha_ltheta *= self.alpha_theta_decay
        self.alpha_fq *= self.alpha_q_decay
        self.alpha_ftheta *= self.alpha_theta_decay

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
        return self.rng.choice(range(self.num_actions), p=self.follower_policies[leader_action])

    def select_committed(self, leader_action):
        """Play the pure strategy that was committed.

        Args:
            leader_action (int): The pure strategy (action) the leader published.

        Returns:
            int: The committed action.

        """
        return leader_action
