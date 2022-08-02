Introduction
=====================

Requirements
-----------------
Ramo requires Python 3.6 or later. It might work with earlier versions but we do not officially support this. We recommend the use of an environment management system such as Conda. This will also allow you to install a specific Python version for Ramo such that it does not conflict with your other Python installations. All other requirements will be installed upon installation of Ramo itself.

.. note::
    Ramo should work on every operating system, but has only been tested on macOS Monterey. If you run intro trouble, please `open an issue <https://github.com/wilrop/ramo/issues>`_.

Installation
-----------------
Installing Ramo is as simple as executing:

.. code-block:: bash

    pip install ramo

This will also include all necessary dependencies.

To install Ramo from scratch, please execute the following steps:

1. Download the latest version of the source code as a ZIP here: https://github.com/wilrop/ramo
2. Unzip the folder in the desired location.
3. Install the necessary dependencies.


Quick Start
-----------------
Getting started with Ramo is as sample as typing:

.. code-block:: Python

    import ramo

A very simple use-case could be to compute a best-response from a multi-objective game and utility function. This can be accomplished by doing the following:

.. code-block:: Python

    import numpy as np

    from ramo.game.monfgs import get_monfg
    from ramo.utility_function.functions import get_u
    from ramo.best_response.best_response import calc_best_response

    game = get_monfg('game1')  # Get a predefined game.
    u = get_u('u1')  # Get a utility function.
    player = 0  # Player zero.
    payoff_matrix = game[player]  # Get this player's payoff matrix.
    player_strategy = np.array([0, 0, 1])  # This strategy will be optimised. The starting point does actually not matter.
    opponent_strategy = np.array([1, 0, 0])  # Define an opponent strategy, here a pure strategy playing only action 1.
    joint_strategy = [player_strategy, opponent_strategy]  # A joint strategy.

    best_response = calc_best_response(u, player, payoff_matrix, joint_strategy)
    print(best_response)

The output at this point will look something like :code:`[1.00000000e+00 3.16413562e-15 0.00000000e+00]`, indicating that playing the first action is a pure strategy best response.
For more in-depth examples, take a look at :doc:`the examples <../examples>`.

Citation
-----------------
To cite the usage of this repository please use the following:

.. code-block:: bibtex

    @misc{ropke2022ramo,
      author = {Willem Röpke},
      title = {Ramo: Rational Agents with Multiple Objectives},
      year = {2022},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/wilrop/mo-game-theory}},
    }

This repository contains work which has first appeared in other publications. Please check out the `list of references <https://github.com/wilrop/ramo/blob/main/references.md>`_ for a complete overview.

Contributing
-----------------
We are building a library containing cutting edge research in multi-objective game theory. If you are working in this area and want to get involved, contributions are very welcome! Our focus is specifically on multi-objective games, but single-objective contributions are also welcome. If you are considering contributing, please send me a message (willem.ropke@vub.be) so we can discuss how to move forward.