Examples
=====================
Ramo allows game theory researchers to quickly verify novel ideas, run experiments and visualise the results. Due to its size, getting started with Ramo may seem daunting. Below, we will present and work out three concrete examples that display a range of possible use cases.

.. note::
    If you believe that a critical use case is not covered in these examples, please do not hesitate to `open an issue <https://github.com/wilrop/ramo/issues>`_.

Example 1: Solving a game
-----------------
The most basic use for Ramo is to solve a game. Solving a game might entail computing a sample equilibrium, retrieving all equilibria or letting us know that no equilibrium exists. Some of these questions are still active areas of research and can as such not be easily answered. However, we can (partially) address most of them directly from within Ramo.

To begin, let us import a predefined game and utility functions.

.. code-block:: Python

    from ramo.game.monfgs import get_monfg
    from ramo.utility_function.functions import get_u

    game = get_monfg('game1')  # Get a predefined game.
    u1, u2 = get_u('u1'), get_u('u1')  # Get a utility function.

Due to Ramo's printing module, we can visualise the payoffs of this game to get a better idea of its structure. We can do this by simply typing:

.. code-block:: Python

    import ramo.utils.printing as pt
    pt.print_monfg(game, 'Game 1')

This results in a game matrix that should look like this:

.. image:: ../images/game.svg

Ramo comes with a collection of games and utility functions that are frequently used in research. In this case, the two utility functions are strictly convex, meaning that we can apply the *MOQUPS* algorithm to compute the full set of pure strategy Nash equilibria in this game. We can do that by making a tuple of all the utility functions and calling the *MOQUPS* algorithm on the game and utility functions.

.. code-block:: Python

    import ramo.best_response.execute_algorithm as ea
    u_tpl = (u1, u2)
    psne = ea.execute_algorithm(game, u_tpl, algorithm='MOQUPS')
    print(psne)

This prints the list :code:`[[array([1., 0., 0.]), array([1., 0., 0.])], [array([0., 0., 1.]), array([0., 0., 1.])]]`. Again, we can visualise this by utilising Ramo's printing functionality:

.. code-block:: Python

    from ramo.utils.strategies import make_profile_from_pure_joint_strat
    action_profiles = [make_profile_from_pure_joint_strat(joint_strat) for joint_strat in psne]
    pt.print_monfg(game, 'Game 1', highlight_cells=action_profiles)

This draws the same game as before but highlights the Nash equilibria in the game.

.. image:: ../images/game-ne.svg

We can use exactly the same setup as described here to find a sample mixed strategy Nash equilibrium using either Fictitious Play (FP) or Iterated Best Response (IBR). It's as simply as setting a different algorithm!

.. code-block:: Python

    ne_fp = ea.execute_algorithm(game, u_tpl, algorithm='FP')
    ne_ibr = ea.execute_algorithm(game, u_tpl, algorithm='IBR')


Example 2: Running baseline algorithms
-----------------
When doing research or writing papers, it is often important to benchmark your algorithms to existing state of the art. This process is made tedious by a lack of standard baseline implementations and environments, requiring everyone to come up with their own. Luckily in Ramo, we provide several learning algorithms which work out of the box on all our games and utility functions.

Let's first define your experimental setup. We gan generate some random game from the :code:`game.generators` module. This will generate a game with payoffs randomly drawn from a discrete uniform distribution.

.. code-block:: Python

    from ramo.game.generators import random_monfg
    from ramo.utility_function.functions import get_u

    game = random_monfg(player_actions=(3, 3), num_objectives=2, reward_min_bound=0, reward_max_bound=5)
    u1, u2 = get_u('u1'), get_u('u1')
    u_tpl = (u1, u2)


After having defined your setup, running an experiment with one of the algorithms is as simply as defining the parameters and calling the executor:

.. code-block:: Python

    from ramo.learner.execute_learner import execute_learner

    experiment = 'indep_ac'  # Independent actor critic.
    runs = 2
    episodes = 10
    rollouts = 10

    data = execute_learner(game, u_tpl, experiment=experiment, runs=runs, episodes=episodes, rollouts=rollouts)
    returns_log, action_probs_log, state_dist_log, metadata = data

Experiments on simple learners return a tuple of four items that can be used in visualisations of the learning process. The first item is a log of the vector valued returns all players received throughout the experiment. The second item shows the evolution of the strategies that players learned. The third item is the joint state distribution. This can for example be used to visualise the states players opted for in distinct stages of the learning process. Lastly, metadata is returned so that experiments can be reproduced easily.

To make this process as interoperable as possible, the exact same setup can be used for experiments which make use of commitment. Commitment is a method where one player commits to playing something in the following round, letting the other player condition their response on this commitment.

.. note::
    Commitment learners are only designed for two-player games.

Below, we show an example where we make use of the non-stationary learning algorithm. This algorithm allows followers to learn a best-response strategy to mixed strategies of the leader. Note that we set :code:`alternate = False`, meaning that we are playing a pure commitment game where one player is the leader in all iterations and the other remains the follower.

.. code-block:: Python

    from ramo.commitment.execute_commitment import execute_commitment

    experiment = 'non_stationary'
    runs = 2
    episodes = 10
    rollouts = 10
    alternate = False

    data = execute_commitment(game, u_tpl, experiment=experiment, runs=runs, episodes=episodes, rollouts=rollouts, alternate=alternate)
    returns_log, action_probs_log, state_dist_log, com_probs_log, metadata = data

Example 3: Hypothesis testing
-----------------
We've now shown some of the most basic use cases that will be useful. However, one of the main selling points of Ramo is the fact that it is a full API. It allows you to pick and choose useful parts in order to test some hypothesis that you have. Below we'll run you through an example.

