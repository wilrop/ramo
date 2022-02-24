import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.nn import softmax

from mo_gt.utils.experiments import softmax_policy


class IndependentActorCriticAgent:
    """An independent learner using the multi-objective actor-critic algorithm for the SER criterion."""

    def __init__(self, u, num_actions, num_objectives, alpha_q=0.01, alpha_theta=0.01, alpha_q_decay=1,
                 alpha_theta_decay=1, rng=None):
        self.u = u
        self.rng = rng if rng is not None else np.random.default_rng()
        self.grad = jit(grad(self.objective_function))
        self.num_actions = num_actions
        self.num_objectives = num_objectives

        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.alpha_q_decay = alpha_q_decay
        self.alpha_theta_decay = alpha_theta_decay

        self.q_table = np.zeros((num_actions, num_objectives))
        self.theta = np.zeros(num_actions)
        self.policy = softmax_policy(self.theta)

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

    def update(self, action, reward):
        """Perform an update for the agent.

        Args:
          action: The actions that was taken by the agent.
          reward: The reward that was obtained by the agent.

        Returns:

        """
        self.update_q_table(action, reward)
        self.theta += self.alpha_theta * self.grad(self.theta, self.q_table)
        self.policy = softmax_policy(self.theta)
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
        """Update the hyperparameters. Decays the learning rate for the Q-values and policy parameters."""
        self.alpha_q *= self.alpha_q_decay
        self.alpha_theta *= self.alpha_theta_decay

    def select_action(self):
        """Select an action according to the agent's policy.

        Returns:
            int: The selected action.

        """
        return self.rng.choice(range(self.num_actions), p=self.policy)
