import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.nn import softmax

from mo_gt.best_response.best_response import calc_expected_returns
from mo_gt.utils.experiments import array_slice, make_strat_from_action


class CoopActionAgent:
    """An agent that optimises a single optimal policy from pure strategy commitment.

    This is mostly intended to be used in an alternating Stackelberg setting, such that players optimise a single
    optimal joint policy.

    """

    def __init__(self, id, u, num_actions, num_objectives, alpha_q=0.01, alpha_theta=0.01, alpha_q_decay=1,
                 alpha_theta_decay=1):
        self.id = id
        self.u = u
        self.grad = jit(grad(self.objective_function))
        self.num_actions = num_actions
        self.num_objectives = num_objectives

        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.alpha_q_decay = alpha_q_decay
        self.alpha_theta_decay = alpha_theta_decay

        self.q_table = np.zeros((num_actions, num_actions, num_objectives))
        self.theta = np.zeros(num_actions)
        self.policy = self.update_policy(self.theta)
        self.last_op_commitment = np.full(num_actions, 1/num_actions)
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
        self.last_op_commitment = make_strat_from_action(commitment, self.num_actions)

        theta = self.next_thetas[commitment]
        joint_policy = [self.last_op_commitment]
        joint_policy.insert(self.id, self.policy)
        q_vals = calc_expected_returns(self.id, self.q_table, joint_policy)

        self.theta += self.alpha_theta * self.grad(theta, q_vals)
        self.policy = self.update_policy(theta)
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

    def update_policy(self, theta):
        """Determine a policy from given parameters.

        Args:
          theta (ndarray): The updated theta parameters.

        Returns:
          ndarray: The updated policy.

        """
        policy = np.asarray(softmax(theta), dtype=float)
        policy = policy / np.sum(policy)
        return policy

    def pre_update_policies(self):
        """Perform a pre update of all policies depending on what commitment is received."""
        for i in range(self.num_actions):
            q_vals = array_slice(self.q_table, self.id, i, i + 1).reshape(self.num_actions, self.num_objectives)
            theta = self.theta + self.alpha_theta * self.grad(self.theta, q_vals)
            policy = self.update_policy(theta)
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
        return np.random.choice(range(self.num_actions), p=self.policy)

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
        return np.random.choice(range(self.num_actions), p=policy)

    def select_committed(self, leader_action):
        """Play the pure strategy that was committed.

        Args:
          leader_action (int): The pure strategy (action) the leader committed to.

        Returns:
          int: The committed action.

        """
        return leader_action
