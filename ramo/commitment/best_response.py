import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.nn import softmax

from ramo.strategy.best_response import calc_best_response
from ramo.strategy.operations import make_joint_strat
from ramo.strategy.strategies import softmax_strategy


class BestResponseAgent:
    """A learner used in two-player Stackelberg games. The leader uses multi-objective actor-critic and the follower
    calculates a best-response using optimisation for the SER.
    """

    def __init__(self, id, u, num_actions, num_objectives, alpha_q=0.01, alpha_theta=0.01, alpha_q_decay=1,
                 alpha_theta_decay=1, epsilon=1, epsilon_decay=0.995, min_epsilon=0.1, rng=None):
        self.id = id
        self.u = u
        self.rng = rng if rng is not None else np.random.default_rng()
        self.grad = jit(grad(self.objective_function))
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.leader_u = np.sum  # Default leader utility function is a simple sum of objectives.

        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.alpha_q_decay = alpha_q_decay
        self.alpha_theta_decay = alpha_theta_decay
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.payoffs_table = np.zeros((num_actions, num_actions, num_objectives))
        self.leader_q_table = np.zeros((num_actions, num_objectives))
        self.leader_theta = np.zeros(num_actions)
        self.leader_policy = softmax_strategy(self.leader_theta)
        self.best_response_policy = np.full(num_actions, 1 / num_actions)

        self.leader = False
        self.calculated = False

    def objective_function(self, theta, q_values):
        """The objective function for the leader. This is the SER criterion.

        Args:
            theta (ndarray): The policy parameters.
            q_values (ndarray): The expected returns for the actions.

        Returns:
            float: The utility from the current policy and leader Q-values.

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

    def set_leader_utility(self, leader_u):
        """Set the leader's utility function. This is used by a pessimistic follower.

        Args:
            leader_u (callable): The utility function used by the leader.

        Returns:

        """
        self.leader_u = leader_u

    def update(self, commitment, actions, reward):
        """Perform an update of the agent. Specifically updates the Q-tables, policies and hyperparameters.

        Args:
            commitment (ndarray): The opponent's committed policy. Unused at this point in time. Still provided to make
                it compatible with other commitment agents.
            actions (List[int]): The actions selected in an episode.
            reward (float): The reward that was obtained by the agent in that episode.

        Returns:

        """
        own_action = actions[self.id]
        self.update_payoffs_table(actions, reward)

        if self.leader:
            self.update_leader_q_table(own_action, reward)
            self.leader_theta += self.alpha_theta * self.grad(self.leader_theta, self.leader_q_table)
            self.leader_policy = softmax_policy(self.leader_theta)

        self.update_parameters()
        self.calculated = False

    def update_leader_q_table(self, own_action, reward):
        """Update the leader's Q-table based on their own action and the obtained reward.

        Args:
            own_action (int): The action taken by the leader.
            reward (float): The reward obtained by this agent.

        Returns:

        """
        self.leader_q_table[own_action] += self.alpha_q * (reward - self.leader_q_table[own_action])

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
        if self.leader:
            self.alpha_theta *= self.alpha_theta_decay
        else:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_commitment(self):
        """Get the commitment from the leader.

        Returns:
            ndarray: The current policy of the leader.

        """
        return self.leader_policy

    def select_action(self, commitment):
        """This method will select an action based on the commitment from the leader.

        Args:
            commitment (ndarray): The commitment from the leader.

        Returns:
            int: The selected action.

        """
        if self.leader:
            return self.select_committed()  # If this agent is the leader, they must follow through.
        else:
            return self.select_counter_action(commitment)  # Otherwise select a counter action.

    def select_counter_action(self, commitment, optimistic=False):
        """Calculate a best-response policy and sample an action from this policy as response to the commitment.

        Args:
            commitment (ndarray): The commitment from the leader.
            optimistic (bool, optional): Whether the agent is optimistic or pessimistic. A pessimistic agent will
                minimise the leader's utility. An optimistic agent will maximise their own utility.
                (Default value = False)

        Returns:
            int: The selected action.

        """
        if not self.calculated:
            strategy = np.full(self.num_actions, 1 / self.num_actions)
            joint_strategy = make_joint_strat(self.id, strategy, [commitment])
            if optimistic:
                self.best_response_policy = calc_best_response(self.u, self.id, self.payoffs_table, joint_strategy)
            else:
                proxy_u = lambda x: - self.leader_u(x)
                self.best_response_policy = calc_best_response(proxy_u, self.id, self.payoffs_table, joint_strategy)
            self.calculated = True

        if self.rng.uniform(0, 1) < self.epsilon:
            return self.rng.integers(self.num_actions)
        else:
            return self.rng.choice(range(self.num_actions), p=self.best_response_policy)

    def select_committed(self):
        """Play an action according to the committed policy.

        Returns:
            int: The selected action.

        """
        return self.rng.choice(range(self.num_actions), p=self.leader_policy)
