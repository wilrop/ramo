class MONFG:
    """A wrapper class for an MONFG.
    """

    def __init__(self, payoffs, players=None):
        self.payoffs = payoffs
        self.num_players = len(payoffs)
        self.players = players if players is not None else list(range(self.num_players))
        self.num_objectives = [self.payoffs[player].shape[-1] for player in range(self.num_players)]
        self.player_actions = self.payoffs[0].shape[:-1]

        if players is not None and len(players) != self.num_players:
            raise Exception('The player list that was provided does not have the correct size')

    def get_num_objectives(self, player=0):
        """Get the number of objectives for a given player.

        Args:
            player (int, optional): The player to get the number of objectives for. (Default value = 0)

        Returns:
            int: The number of objectives in the game for the given player.
        """
        return self.num_objectives[player]

    def get_payoff_matrix(self, player):
        """Get the payoff matrix for a specific player.

        Args:
            player (int): The player index.

        Returns:
            ndarray: A payoff matrix.
        """
        return self.payoffs[player]
