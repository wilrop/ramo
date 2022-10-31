import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.nn import softmax

from ramo.strategy.best_response import calc_expected_returns
from ramo.strategy.operations import make_joint_strat
from ramo.strategy.strategies import softmax_strategy


class CoopPolicyAgent:
    """An agent that optimises a single optimal policy from mixed strategy commitment [1].

    References:
        .. [1] Willem Röpke, Diederik M. Roijers, Ann Nowé, & Roxana Rădulescu. (2021). Preference Communication in
            Multi-Objective Normal-Form Games.

    """

    def __init__(self, id, u, num_actions, num_objectives, alpha_q=0.01, alpha_theta=0.01, alpha_q_decay=1,
                 alpha_theta_decay=1, rng=None):
        self.id = id
        self.u = u
        self.rng = rng if rng is not None else np.random.default_rng()
        self.grad = jit(grad(self.objective_function))
        self.num_actions = num_actions
        self.num_objectives = num_objectives

        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.alpha_q_decay = alpha_q_decay
        self.alpha_theta_decay = alpha_theta_decay

        self.q_table = np.zeros((num_actions, num_actions, num_objectives))
        self.theta = np.zeros(num_actions)
        self.policy = softmax_strategy(self.theta)
        self.leader_policy = np.full(num_actions, 1 / num_actions)

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
            commitment (int): The opponent's committed action.
            actions (List[int]): The actions selected in an episode.
            reward (float): The reward that was obtained by the agent in that episode.

        Returns:

        """
        if not self.leader:
            # Perform pre-commitment update.
            joint_policy = make_joint_strat(self.id, self.policy, [self.leader_policy])
            q_vals = calc_expected_returns(self.id, self.q_table, joint_policy)

            self.theta += self.alpha_theta * self.grad(self.theta, q_vals)
            self.policy = softmax_policy(self.theta)

            self.leader_policy = commitment  # Update leader policy from the commitment.

        self.update_q_table(actions, reward)

        # Perform post-commitment update.
        joint_policy = make_joint_strat(self.id, self.policy, [self.leader_policy])
        q_vals = calc_expected_returns(self.id, self.q_table, joint_policy)

        self.theta += self.alpha_theta * self.grad(self.theta, q_vals)
        self.policy = softmax_policy(self.theta)

        self.update_parameters()

    def update_q_table(self, actions, reward):
        """Update the joint-action Q-table.

        Args:
            actions (List[int]): The actions taken by the agents.
            reward (float): The reward obtained by this agent.

        Returns:

        """
        self.q_table[actions[0], actions[1]] += self.alpha_q * (reward - self.q_table[actions[0], actions[1]])

    def update_parameters(self):
        """Update the internal parameters of the agent."""
        self.alpha_q *= self.alpha_q_decay
        self.alpha_theta *= self.alpha_theta_decay

    def get_commitment(self):
        """Get the commitment from the leader.

        Returns:
            ndarray: The current strategy of the leader.

        """
        return self.policy

    def select_action(self, commitment):
        """Select an action based on the commitment of the leader.

        Args:
            commitment (ndarray): The message that was sent.

        Returns:
            int: The selected action.

        """
        if self.leader:
            return self.select_committed(commitment)  # If this agent is committing, they must follow through.
        else:
            return self.select_counter_action(commitment)  # Otherwise select a counter action.

    def select_counter_action(self, leader_strategy):
        """Perform an update to learn a counter policy and sample an action using this policy.

        Args:
            leader_strategy (ndarray): The committed pure strategy from the leader.

        Returns:
            int: The selected action.

        """
        joint_policy = make_joint_strat(self.id, self.policy, [leader_strategy])
        q_vals = calc_expected_returns(self.id, self.q_table, joint_policy)
        theta = self.theta + self.alpha_theta * self.grad(self.theta, q_vals)
        policy = softmax_policy(theta)
        return self.rng.choice(range(self.num_actions), p=policy)

    def select_committed(self, leader_strategy):
        """Sample an action from the committed strategy.

        Args:
            leader_strategy (ndarray): The mixed strategy the leader committed to.

        Returns:
            int: The committed action.

        """
        return self.rng.choice(range(self.num_actions), p=leader_strategy)
