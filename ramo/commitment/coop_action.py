import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.nn import softmax

from ramo.strategy.best_response import calc_expected_returns
from ramo.strategy.operations import make_joint_strat, make_strat_from_action
from ramo.strategy.strategies import softmax_strategy
from ramo.utils.helpers import array_slice


class CoopActionAgent:
    """An agent that optimises a single optimal policy from pure strategy commitment [1].

    This is mostly intended to be used in an alternating Stackelberg setting, such that players optimise a single
    optimal joint policy.

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
        self.last_op_commitment = np.full(num_actions, 1 / num_actions)
        self.next_thetas = np.zeros((num_actions, num_actions))
        self.next_policies = np.tile(self.policy, (num_actions, 1))

        self.leader = False
        self.calculated = False

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
        self.update_q_table(actions, reward)

        if not self.leader:
            self.last_op_commitment = make_strat_from_action(commitment, self.num_actions)

        theta = self.next_thetas[commitment]
        joint_policy = make_joint_strat(self.id, self.policy, [self.last_op_commitment])
        q_vals = calc_expected_returns(self.id, self.q_table, joint_policy)

        self.theta += self.alpha_theta * self.grad(theta, q_vals)
        self.policy = softmax_policy(self.theta)
        self.next_thetas = np.tile(self.theta, (self.num_actions, 1))
        self.next_policies = np.tile(self.policy, (self.num_actions, 1))

        self.update_parameters()
        self.calculated = False

    def update_q_table(self, actions, reward):
        """Update the joint-action Q-table.

        Args:
            actions (List[int]): The actions taken by the agents.
            reward (float): The reward obtained by this agent.

        Returns:

        """
        idx = tuple(actions)
        self.q_table[idx] += self.alpha_q * (reward - self.q_table[idx])

    def pre_update_policies(self):
        """Perform a pre update of all policies depending on what commitment is received."""
        for i in range(self.num_actions):
            q_vals = array_slice(self.q_table, self.id, i, i + 1).reshape(self.num_actions, self.num_objectives)
            theta = self.theta + self.alpha_theta * self.grad(self.theta, q_vals)
            policy = softmax_policy(theta)
            self.next_thetas[i] = theta
            self.next_policies[i] = policy

    def update_parameters(self):
        """Update the internal parameters of the agent."""
        self.alpha_q *= self.alpha_q_decay
        self.alpha_theta *= self.alpha_theta_decay

    def get_commitment(self):
        """Get the commitment from the leader.

        Returns:
            int: A pure strategy commitment of the leader.

        """
        return self.rng.choice(range(self.num_actions), p=self.policy)

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
        if not self.calculated:
            self.pre_update_policies()
            self.calculated = True
        policy = self.next_policies[leader_action]
        return self.rng.choice(range(self.num_actions), p=policy)

    def select_committed(self, leader_action):
        """Play the pure strategy that was committed.

        Args:
            leader_action (int): The pure strategy (action) the leader committed to.

        Returns:
            int: The committed action.

        """
        return leader_action
