def get_player_actions(monfg):
    """Get the number of actions per player for a given MONFG.

    Args:
        monfg (List[ndarray]): An MONFG as a list of payoff matrices.

    Returns:
        Tuple[int]: A tuple with the number of actions per player.
    """
    return monfg[0].shape[:-1]


def get_num_objectives(monfg, individual=False):
    """

    Args:
        monfg (List[ndarray]): An MONFG as a list of payoff matrices.
        individual (bool, optional): Whether to assume players have the same number of objectives.
        (Default value = False)

    Returns:
        int | Tuple[int]: An integer with the number of objectives or a tuple with the number of objectives per player.
    """
    if individual:
        return tuple([payoff_matrix.shape[-1] for payoff_matrix in monfg])
    else:
        return monfg[0].shape[-1]
