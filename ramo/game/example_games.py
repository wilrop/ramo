import numpy as np

from ramo.game.monfg import MONFG

payoffs1 = [np.array([[(4, 0), (3, 1), (2, 2)],
                      [(3, 1), (2, 2), (1, 3)],
                      [(2, 2), (1, 3), (0, 4)]], dtype=float),
            np.array([[(4, 0), (3, 1), (2, 2)],
                      [(3, 1), (2, 2), (1, 3)],
                      [(2, 2), (1, 3), (0, 4)]], dtype=float)]
"""List[ndarray]: Game 1 - The (im)balancing act game. A 3-action 2-player game with team rewards.

There are no NE under SER using u1 and u2.
There are two PSNE using u1 and u5: [0, 0], [2, 2]. Checked for correctness using Gambit.
"""

payoffs2 = [np.array([[(4, 0), (2, 2)],
                      [(2, 2), (0, 4)]], dtype=float),
            np.array([[(4, 0), (2, 2)],
                      [(2, 2), (0, 4)]], dtype=float)]
"""List[ndarray]: Game 2 - The (im)balancing act game without M. A 2-action 2-player game with team rewards.

There are no NE under SER using u1 and u2.
There are two PSNE using u1 and u5: [0, 0], [1, 1]. Checked for correctness using Gambit.
"""

payoffs3 = [np.array([[(4, 0), (3, 1)],
                      [(3, 1), (2, 2)]], dtype=float),
            np.array([[(4, 0), (3, 1)],
                      [(3, 1), (2, 2)]], dtype=float)]
"""List[ndarray]: Game 3 - The (im)balancing act game without R. A 2-action 2-player game with team rewards.

There is one NE under SER using u1 and u2: (L, M).
There is one PSNE using u1 and u5: [0, 0]. Checked for correctness using Gambit.
"""

payoffs4 = [np.array([[(4, 1), (1, 2)],
                      [(3, 1), (3, 2)]], dtype=float),
            np.array([[(4, 1), (1, 2)],
                      [(3, 1), (3, 2)]], dtype=float)]
"""List[ndarray]: Game 4 - A 2-action 2-player game with team rewards.

There is one NE under SER using u1 and u2: (M, M).
Several papers have claimed that (L,L) is also a NE. This is false. Player 2 has an incentive to deviate to (5/6, 1/6).
There are two PSNE using u1 and u5: [0, 0], [1, 1]. Checked for correctness using Gambit.
This game shows cyclic behaviour under IBR with simultaneous updates but not with alternating updates.
This game shows no cyclic behaviour with fictitious play.
"""

payoffs5 = [np.array([[(4, 1), (1, 2), (2, 1)],
                      [(3, 1), (3, 2), (1, 2)],
                      [(1, 2), (2, 1), (1, 3)]], dtype=float),
            np.array([[(4, 1), (1, 2), (2, 1)],
                      [(3, 1), (3, 2), (1, 2)],
                      [(1, 2), (2, 1), (1, 3)]], dtype=float)]
"""List[ndarray]: Game 5 - A 3-action 2-player game with team rewards.

There is one NE under SER using u1 and u2: (M, M).
Several papers have claimed that (L,L) and (R, R) are also NE. This is false. Player 2 has an incentive to deviate in 
both cases. For (L, L) they deviate to (5/6, 1/6, 0) and for (R, R) they deviate to (0, 1/4, 3/4).
There are three NE under SER using u1 and u2: (L,L), (M,M) and (R,R).
There are three PSNE using u1 and u5: [0, 0], [1, 1], [2, 2]. Checked for correctness using Gambit.
This game shows cyclic behaviour under IBR with simultaneous updates but not with alternating updates.
"""

payoffs6 = [np.array([[(0, 0), (7, 2)],
                      [(2, 7), (6, 2.32502)]], dtype=float),
            np.array([[(0, 0), (2, 7)],
                      [(7, 2), (6, 2.32502)]], dtype=float)]
"""List[ndarray]: Game 6 - A multi-objectivised version of the game of chicken. Both players use the utility function u2.

The cyclic equilibrium is to go 2/3 your own, 1/3 other action uniformly over these.
"""

payoffs7 = [np.array([[(-1, -1), (-1, 1)],
                      [(1, -1), (1, 1)]], dtype=float),
            np.array([[(-1, -1), (-1, 1)],
                      [(1, -1), (1, 1)]], dtype=float)]
"""List[ndarray]: Game 7 - A 2-action 2-player game with team rewards.

An example of a game where commitment may be exploited.
"""

payoffs8 = [np.array([[(10, 2), (0, 0)],
                      [(0, 0), (2, 10)]], dtype=float),
            np.array([[(10, 2), (0, 0)],
                      [(0, 0), (2, 10)]], dtype=float)]
"""List[ndarray]: Game 8 - A 2-action 2-player game with team rewards.

There are two NE when both agents use utility function u2 under SER: (L,L) and (R, R).
The cyclic equilibrium is to mix uniformly over these.
"""

payoffs9 = [np.array([[(10, 2), (2, 3)],
                      [(4, 2), (6, 3)]], dtype=float),
            np.array([[(10, 2), (2, 3)],
                      [(4, 2), (6, 3)]], dtype=float)]
"""List[ndarray]: Game 9 - A 2-action 2-player game with team rewards.

A noisy version of game 8.
The cyclic equilibrium with utility function u2 is to play A 75% of the time and 25% B.
"""

payoffs10 = [np.array([[(2, 0), (0, 1)],
                       [(1, 0), (0, 2)]], dtype=float),
             np.array([[(2, 0), (1, 1)],
                       [(1, 1), (0, 2)]], dtype=float)]
"""List[ndarray]: Game 10 - A 2-action 2-player game with individual rewards.

This game has no Nash equilibrium with utility functions u1 and u2, but does have a cyclic Nash equilibrium.
"""

payoffs11 = [np.array([[(2, 0), (1, 1)],
                       [(1, 1), (0, 2)]], dtype=float),
             np.array([[(2, 0), (0, 1)],
                       [(1, 0), (0, 2)]], dtype=float)]
"""List[ndarray]: Game 11 - A 2-action 2-player game with individual rewards.

The same game as game 10 but intended to be used with the utility functions reversed.
"""

payoffs12 = [
    np.array([[(4, 1), (1, 2), (2, 1)],
              [(3, 1), (3, 2), (1, 2)],
              [(1, 2), (2, 1), (1, 3)]], dtype=float),
    np.array([[(4, 0), (3, 1), (2, 2)],
              [(3, 1), (2, 2), (1, 3)],
              [(2, 2), (1, 3), (0, 4)]], dtype=float)]
"""List[ndarray]: Game 12 - A 3-action 2-player game with individual rewards.

This game has two PSNE using u1 and u5: [0, 0], [2, 2]. Checked for correctness using Gambit.
"""

payoffs13 = [
    np.array([[(2, 3), (3, 2), (1, 1)],
              [(2, 5), (0, 2), (5, 2)],
              [(1, 3), (4, 0), (1, 3)]], dtype=float),
    np.array([[(0, 3), (1, 2), (2, 1)],
              [(2, 2), (3, 2), (1, 2)],
              [(3, 1), (0, 3), (1, 0)]], dtype=float)]
"""List[ndarray]: Game 13 - A 3-action 2-player game with individual rewards.

This game has no PSNE using u1 and u5. Checked for correctness using Gambit.
"""

payoffs14 = [
    np.array([[[(1, 0), (2, 1)],
               [(3, 0), (1, 2)]],
              [[(0, 2), (2, 2)],
               [(3, 1), (2, 0)]]], dtype=float),
    np.array([[[(2, 0), (0, 2)],
               [(1, 1), (1, 2)]],
              [[(0, 0), (1, 2)],
               [(2, 1), (0, 0)]]], dtype=float),
    np.array([[[(1, 2), (2, 1)],
               [(0, 1), (2, 2)]],
              [[(1, 1), (0, 3)],
               [(1, 1), (1, 2)]]], dtype=float)]
"""List[ndarray]: Game 14 - A 2-action 3-player game with individual rewards.

This game has two PSNE using u1, u5 and u6: [0, 1, 1], [1, 0, 1]. Checked for correctness by hand.
"""

payoffs15 = [
    np.array([[[(1, 0), (2, 1), (1, 2)],
               [(3, 0), (1, 2), (2, 2)]],
              [[(0, 2), (2, 2), (3, 0)],
               [(3, 1), (2, 0), (0, 1)]],
              [[(1, 1), (0, 0), (2, 1)],
               [(1, 2), (2, 0), (3, 0)]]], dtype=float),
    np.array([[[(0, 2), (0, 1), (1, 1)],
               [(1, 3), (2, 2), (2, 2)]],
              [[(0, 2), (2, 0), (3, 0)],
               [(3, 1), (1, 0), (2, 1)]],
              [[(2, 2), (2, 1), (2, 0)],
               [(0, 1), (1, 3), (1, 1)]]], dtype=float),
    np.array([[[(1, 3), (1, 1), (2, 2)],
               [(2, 1), (2, 3), (2, 0)]],
              [[(0, 2), (1, 1), (3, 1)],
               [(3, 1), (2, 1), (2, 1)]],
              [[(0, 1), (1, 0), (0, 0)],
               [(1, 1), (2, 1), (1, 1)]]], dtype=float)]
"""List[ndarray]: Game 15 - A 3-player game where p1 has 3 actions, p2 has 2 and p3 has 3, with individual rewards.

This game has three PSNE using u1, u5 and u6: [0, 1, 1], [1, 0, 2], [1, 1, 0]. Checked for correctness by hand.
"""

payoffs16 = [
    np.array([[(2, 0), (1, 0)],
              [(0, 1), (0, 2)]], dtype=float),
    np.array([[(1, 0), (0, 2)],
              [(2, 0), (0, 1)]], dtype=float)]
"""List[ndarray]: Game 16 - A 2-player 2-action game with individual rewards.

This game has no NE when both players use the utility function u1.
"""

payoffs17 = [np.array([[(4, 1), (1, 1.5)],
                       [(3, 1), (3, 2)]], dtype=float),
             np.array([[(4, 1), (1, 1.5)],
                       [(3, 1), (3, 2)]], dtype=float)]
"""List[ndarray]: Game 17 - A 2-action 2-player game with team rewards.

There are two NE under SER using u1 and u2: (L, L) and (M, M).
This game solves the problem from game 4, which in some papers falsely claims to have these two strategies as NE.
There are two PSNE using u1 and u5: [0, 0], [1, 1]. Checked for correctness using Gambit.
This game shows cyclic behaviour under IBR with simultaneous updates but not with alternating updates.
This game shows no cyclic behaviour with fictitious play.
"""

payoffs18 = [np.array([[(4, 1), (1, 1.5), (2, 1)],
                       [(3, 1), (3, 2), (1, 2)],
                       [(1, 2), (2, 1.5), (1.5, 3)]], dtype=float),
             np.array([[(4, 1), (1, 1.5), (2, 1)],
                       [(3, 1), (3, 2), (1, 2)],
                       [(1, 2), (2, 1.5), (1.5, 3)]], dtype=float)]
"""List[ndarray]: Game 18 - A 3-action 2-player game with team rewards.

There are three NE under SER using u1 and u2: (M, M), (L, L) and (R, R).
This game solves the problem from game 5, which in some papers falsely claims to have these three strategies as NE.
There are three NE under SER using u1 and u2: (L,L), (M,M) and (R,R).
There are three PSNE using u1 and u5: [0, 0], [1, 1], [2, 2]. Checked for correctness using Gambit.
This game shows cyclic behaviour under IBR with simultaneous updates but not with alternating updates.
"""

payoffs19 = [np.array([[(7, 12), (3, 4)],
                       [(11, 7), (7, 3)]], dtype=float),
             np.array([[(6, 2), (5, 4)],
                       [(2, 8), (7, 2)]], dtype=float)]
"""List[ndarray]: Game 19 - A 2-action 2-player game with individual rewards.

This game has a NE of {(0.5, 0.5), (0.75, 0.25)} using u2 for both players. 
It is used to test the correctness of the FP and IBR algorithms.
"""


def get_monfg(game):
    """Get a predefined MONFG.

    Args:
        game (str): The string of the game.

    Returns:
        MONFG: An MONFG object initiated with predefined payoffs.
    """
    if game == 'game1':
        payoffs = payoffs1
    elif game == 'game2':
        payoffs = payoffs2
    elif game == 'game3':
        payoffs = payoffs3
    elif game == 'game4':
        payoffs = payoffs4
    elif game == 'game5':
        payoffs = payoffs5
    elif game == 'game6':
        payoffs = payoffs6
    elif game == 'game7':
        payoffs = payoffs7
    elif game == 'game8':
        payoffs = payoffs8
    elif game == 'game9':
        payoffs = payoffs9
    elif game == 'game10':
        payoffs = payoffs10
    elif game == 'game11':
        payoffs = payoffs11
    elif game == 'game12':
        payoffs = payoffs12
    elif game == 'game13':
        payoffs = payoffs13
    elif game == 'game14':
        payoffs = payoffs14
    elif game == 'game15':
        payoffs = payoffs15
    elif game == 'game16':
        payoffs = payoffs16
    elif game == 'game17':
        payoffs = payoffs17
    elif game == 'game18':
        payoffs = payoffs18
    elif game == 'game19':
        payoffs = payoffs19
    else:
        raise Exception("The provided game does not exist.")

    return MONFG(payoffs)
