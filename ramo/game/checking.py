import numpy as np


def is_degenerate_pure(game):
    """Check if pure strategies are different for each opponent joint strategy.

    Args:
        game (List[ndarray]): An input MONFG as a list of payoff matrices.

    Returns:
        bool: Whether the game is degenerate in pure strategies.
    """
    player_actions = game[0].shape[:-1]

    for player, (payoff_matrix, num_actions) in enumerate(zip(game, player_actions)):
        opp_actions = tuple(np.delete(player_actions, player))

        for opp_strat in np.ndindex(*opp_actions):
            slice_idx = opp_strat[:player] + (slice(0, num_actions),) + opp_strat[player:]
            payoffs = payoff_matrix[slice_idx]
            unique_count = len(np.unique(payoffs, axis=0))

            if unique_count != num_actions:
                return True
    return False
