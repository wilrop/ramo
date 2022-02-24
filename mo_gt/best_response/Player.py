import numpy as np

from mo_gt.best_response.best_response import calc_best_response


class IBRPlayer:
    """A player that learns a strategy using best-response iteration."""

    def __init__(self, id, u, num_actions, payoff_matrix, init_strategy=None, rng=None):
        self.id = id
        self.u = u
        self.num_actions = num_actions
        self.payoff_matrix = payoff_matrix
        self.rng = rng if rng is not None else np.random.default_rng()
        if init_strategy is None:
            self.strategy = np.full(self.num_actions, 1 / self.num_actions)
        else:
            self.strategy = init_strategy

    def update_strategy(self, joint_strategy):
        """Update the strategy by calculating a best response to the other players' strategies.

        Args:
          joint_strategy (List[ndarray]): A list of each player's individual strategy.

        Returns:
          Tuple(bool, ndarray): Whether the strategy has converged and the best response strategy.

        """
        br = calc_best_response(self.u, self.id, self.payoff_matrix, joint_strategy)
        if (br == self.strategy).all():
            done = True
        else:
            done = False
        self.strategy = br
        return done, br


class FPPlayer:
    """A player that learns a strategy using the fictitious play algorithm."""

    def __init__(self, id, u, player_actions, payoff_matrix, init_strategy=None, rng=None):
        self.id = id
        self.u = u
        self.player_actions = player_actions
        self.num_actions = player_actions[id]
        self.payoff_matrix = payoff_matrix
        self.rng = rng if rng is not None else np.random.default_rng()
        if init_strategy is None:
            self.strategy = np.full(self.num_actions, 1 / self.num_actions)
        else:
            self.strategy = init_strategy
        self.empirical_strategies = [np.zeros(num_actions) for num_actions in player_actions]

    def select_action(self):
        """Select an action using the current strategy.

        Returns:
            int: The selected action.

        """
        return self.rng.choice(range(self.num_actions), p=self.strategy)

    def calc_joint_strategy(self):
        """Calculates the empirical joint strategy.

        Returns:
            List[ndarray]: The joint strategy.

        """
        joint_strategy = []
        for player_actions in self.empirical_strategies:
            strategy = player_actions / np.sum(player_actions)
            joint_strategy.append(strategy)
        return joint_strategy

    def update_strategy(self):
        """Updates the strategy of the player by calculating a best response to the empirical joint strategy.

        Returns:
            Tuple(bool, ndarray): Whether the strategy has converged and the best response strategy.

        """
        joint_strategy = self.calc_joint_strategy()
        br = calc_best_response(self.u, self.id, self.payoff_matrix, joint_strategy)
        if (br == self.strategy).all():
            done = True
        else:
            done = False
        self.strategy = br
        return done, br

    def update_empirical_strategies(self, actions):
        """Update the empirical strategy of all players.

        Args:
          actions (List[int]): The actions that were taken by the players.

        """
        for player, action in enumerate(actions):
            self.empirical_strategies[player][action] += 1
