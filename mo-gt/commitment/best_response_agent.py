import numpy as np
from utils_learn import *
import games
from scipy.optimize import minimize


def objective(strategy, expected_returns, u):
    """
    The objective function to minimise for is the negative SER, as we want to maximise for the SER.
    :param strategy: The current estimate for the best response strategy.
    :param expected_returns: The expected returns given all other players' strategies.
    :param u: The utility function of this agent.
    :return: A best response policy.
    """
    expected_vec = strategy @ expected_returns  # The expected vector of the strategy applied to the expected returns.
    objective = - u(expected_vec)  # The negative utility.
    return objective


def best_response(u, player, payoff_matrix, joint_strategy):
    """
    This function calculates a best response for a given player.
    :param u: The utility function for this player.
    :param player: The player id.
    :param payoff_matrix: The payoff matrix for this player.
    :param joint_strategy: The joint strategy of all players.
    :return: A best response strategy.
    """
    num_objectives = payoff_matrix.shape[-1]
    num_actions = len(joint_strategy[player])
    num_players = len(joint_strategy)
    opponents = np.delete(np.arange(num_players), player)
    expected_returns = payoff_matrix

    for opponent in opponents:  # Loop over all opponent strategies.
        strategy = joint_strategy[opponent]  # Get this opponent's strategy.

        # We reshape this strategy to be able to multiply along the correct axis for weighting expected returns.
        # For example if you end up in [1, 2] or [2, 3] with 50% probability.
        # We calculate the individual expected returns first: [0.5, 1] or [1, 1.5]
        dim_array = np.ones((1, expected_returns.ndim), int).ravel()
        dim_array[opponent] = -1
        strategy_reshaped = strategy.reshape(dim_array)

        expected_returns = expected_returns * strategy_reshaped  # Calculate the probability of a joint state occurring.
        # We now take the sum of the weighted returns to get the expected returns.
        # We need keepdims=True to make sure that the opponent still exists at the correct axis, their action space is
        # just reduced to one action resulting in the expected return now.
        expected_returns = np.sum(expected_returns, axis=opponent, keepdims=True)

    expected_returns = expected_returns.reshape(num_actions, num_objectives)  # Cast the result to a correct shape.

    init_guesses = [np.full(num_actions, 1 / num_actions)]  # A uniform strategy as first guess for the optimiser.
    for i in range(num_actions):
        pure_strat = np.zeros(num_actions)
        pure_strat[i] = 1
        init_guesses.append(pure_strat)  # A pure strategy as first guess for the optimiser.

    best_response = None
    best_utility = float('inf')
    for init_guess in init_guesses:
        bounds = [(0, 1)] * num_actions  # Constrain probabilities to 0 and 1.
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Equality constraint is equal to zero by default.
        res = minimize(lambda x: objective(x, expected_returns, u), init_guess, bounds=bounds, constraints=constraints)

        if res['fun'] < best_utility:
            best_utility = res['fun']
            best_response = res['x'] / np.sum(res['x'])  # In case of floating point errors.

    return best_response


class BestResponseAgent:
    """
    This class represents an agent that uses the SER multi-objective optimisation criterion.
    """

    def __init__(self, id, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives, opt=False, epsilon=1,
                 epsilon_decay=0.995):
        self.id = id
        self.u = u
        self.du = du
        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.alpha_decay = alpha_decay
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        # optimistic initialization of Q-table
        if opt:
            self.msg_q_table = np.ones((num_actions, num_objectives)) * 20
        else:
            self.msg_q_table = np.zeros((num_actions, num_objectives))
        self.payoffs_table = np.zeros((num_actions, num_actions, num_objectives))
        self.msg_theta = np.zeros(num_actions)
        self.msg_policy = np.full(num_actions, 1 / num_actions)
        self.best_response_policy = np.full(num_actions, 1 / num_actions)
        self.communicating = False
        self.calculated = False
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def update(self, communicator, message, actions, reward):
        """
        This method will update the Q-table, strategy and internal parameters of the agent.
        :param communicator: The id of the communicating agent.
        :param message: The message that was sent.
        :param actions: The actions selected in the previous episode.
        :param reward: The reward that was obtained by the agent.
        :return: /
        """
        self.update_payoffs_table(actions, reward)
        own_action = actions[self.id]

        if communicator == self.id:
            self.update_msg_q_table(own_action, reward)
            theta, policy = self.update_policy(self.msg_policy, self.msg_theta, self.msg_q_table)
            self.msg_theta = theta
            self.msg_policy = policy
        else:
            self.epsilon = min(0.1, self.epsilon * self.epsilon_decay)

        self.update_parameters()
        self.calculated = False

    def update_msg_q_table(self, action, reward):
        """
        This method will update the Q-table based on the chosen actions and the obtained reward.
        :param action: The action chosen by this agent.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        self.msg_q_table[action] += self.alpha_q * (reward - self.msg_q_table[action])

    def update_payoffs_table(self, actions, reward):
        """
        This method will update the payoffs table to learn the payoff vector of joint actions.
        :param actions: The actions that were taken in the previous episode.
        :param reward: The reward obtained by this joint action.
        :return: /
        """
        self.payoffs_table[actions[0], actions[1]] += self.alpha_q * (
                reward - self.payoffs_table[actions[0], actions[1]])

    def update_policy(self, policy, theta, expected_q):
        """
        This method will update the given theta parameters and policy.
        :param policy: The policy we want to update.
        :param theta: The current theta parameters for this policy.
        :param expected_q: The Q-values for this policy.
        :return: Updated theta parameters and policy.
        """
        policy = np.copy(policy)  # This avoids some weird numpy bugs where the policy/theta is referenced by pointer.
        theta = np.copy(theta)
        expected_u = policy @ expected_q
        # We apply the chain rule to calculate the gradient.
        grad_u = self.du(expected_u)  # The gradient of u
        grad_pg = softmax_grad(policy).T @ expected_q  # The gradient of the softmax function
        grad_theta = grad_u @ grad_pg.T  # The gradient of the complete function J(theta).
        theta += self.alpha_theta * grad_theta
        policy = softmax(theta)
        return theta, policy

    def update_parameters(self):
        """
        This method will update the internal parameters of the agent.
        :return: /
        """
        self.alpha_q *= self.alpha_decay
        self.alpha_theta *= self.alpha_decay

    def select_action(self, message):
        """
        This method will select an action based on the message that was sent.
        :param message: The message that was sent.
        :return: The selected action.
        """
        if self.communicating:
            self.communicating = False
            return self.select_committed()  # If this agent is committing, they must follow through.
        else:
            return self.select_counter_action(message)  # Otherwise select a counter action.

    def get_message(self):
        """
        This method will determine what action this agent will publish.
        :return: The action that will maximise this agent's SER, given that the other agent also maximises its response.
        """
        self.communicating = True
        return self.msg_policy

    def select_counter_action(self, op_policy, optimistic=False):
        """
        This method will select the best counter policy and choose an action using this policy.
        :param op_policy: The message from an agent in the form of their current policy.
        :param optimistic: Whether the agent is optimistic or pessimistic.
        :return: The selected action.
        """
        if not self.calculated:
            strategy = np.full(self.num_actions, 1 / self.num_actions)
            joint_strategy = [op_policy, strategy]
            if optimistic:
                self.best_response_policy = best_response(self.u, self.id, self.payoffs_table, joint_strategy)
            else:
                proxy_u = lambda x: - games.u3(x)
                self.best_response_policy = best_response(proxy_u, self.id, self.payoffs_table, joint_strategy)
            self.calculated = True

        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.random.choice(range(self.num_actions), p=self.best_response_policy)

    def select_committed(self):
        """
        This method simply plays the action that it already published.
        :return: The action it published.
        """
        return np.random.choice(range(self.num_actions), p=self.msg_policy)
