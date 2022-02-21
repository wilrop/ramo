import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.nn import softmax


class OptionalComAgent:
    """An agent that learns when to commit through a two layer system.

    This is implemented through learning two additional agents, one used when committing and another used when not
    committing.
    """

    def __init__(self, no_com_agent, com_agent, id, u, num_actions, num_objectives, alpha_q=0.01, alpha_theta=0.01, alpha_q_decay=1, alpha_theta_decay=1):
        self.no_com_agent = no_com_agent
        self.com_agent = com_agent

        self.id = id
        self.u = u
        self.grad = jit(grad(self.objective_function))
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.num_options = 2  # Only two types of commitment: commit or don't commit.

        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.alpha_q_decay = alpha_q_decay
        self.alpha_theta_decay = alpha_theta_decay

        self.q_table = np.zeros((self.num_options, num_objectives))
        self.theta = np.zeros(self.num_options)
        self.policy = self.update_policy(self.theta)

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
        """Perform an update of the agent and cascade an update to the lower layer agent.

        Args:
          commitment (int): The leader's non-stationary commitment strategy.
          actions (List[int]): The actions selected in an episode.
          reward (float): The reward that was obtained by the agent in that episode.

        Returns:

        """
        print("id:", self.id)
        self.update_q_table(commitment, reward)

        self.theta += self.alpha_theta * self.grad(self.theta, self.q_table)
        self.policy = self.update_policy(self.theta)

        if commitment is None:
            print("here")
            self.no_com_agent.update(actions[self.id], reward)
        else:
            self.com_agent.update(commitment, actions, reward)

        self.update_parameters()
        print(self.no_com_agent.theta)
        print(self.no_com_agent.policy)
        print(self.no_com_agent.q_table)

    def update_q_table(self, commitment, reward):
        """Update the vector-valued Q-table.

        Args:
            commitment (int | ndarray | None): The commitment from the leader.
            reward (float): The reward obtained by this agent.

        Returns:

        """
        if commitment is None:
            idx = 0
        else:
            idx = 1
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

    def update_parameters(self):
        """Update the internal parameters of the agent."""
        self.alpha_q *= self.alpha_q_decay
        self.alpha_theta *= self.alpha_theta_decay

    def get_commitment(self):
        """Get the commitment from the leader.

        Returns:
            int | None: A commitment from the leader.

        """
        commit = np.random.choice(range(self.num_options), p=self.policy)
        if commit == 0:  # Don't communicate
            return None
        else:
            return self.com_agent.get_commitment()

    def select_action(self, commitment):
        """Select an action based on the commitment of the leader. Pass the commitment to the correct layer.

        Args:
          commitment (int | ndarray): The commitment from the leader.

        Returns:
          int: The selected action.

        """
        if commitment is None:
            return self.no_com_agent.select_action()
        else:
            return self.com_agent.select_action(commitment)
