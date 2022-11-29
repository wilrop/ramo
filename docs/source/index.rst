.. Ramo documentation master file, created by
   sphinx-quickstart on Fri Mar 11 16:06:15 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Ramo
================================
Welcome to the documentation for Ramo: Rational Agents with Multiple Objectives.

Ramo is an algorithmic game theory framework offering a collection of algorithms and utilities for computing or learning (approximate) equilibria in multi-objective games.
As of now, the framework supports only Multi-Objective Normal-Form Games (MONFGs), which are the multi-objective counterpart to normal-form games.

We also provide a number of handy utilities. For example, this repository comes with a number of known and pre-analysed example games and utility functions. There are also helper functions to generate random games, scalarise games, etc. In addition, there is functionality to save and plot data resulting from experiments
and a bunch more, with even more on the way!


.. toctree::
   :maxdepth: 1
   :caption: User Guide

   Introduction <introduction.rst>
   Examples <examples.rst>


.. toctree::
   :maxdepth: 1
   :caption: Algorithms

   Overview <algorithms.rst>
   Nash Equilibria <ramo.nash.rst>
   Pareto <ramo.pareto.rst>
   Best Response <ramo.game.rst>
   Learners <ramo.learner.rst>
   Commitment <ramo.commitment.rst>

.. toctree::
   :maxdepth: 1
   :caption: Utilities

   Overview <modules.rst>
   Game <ramo.game.rst>
   Strategy <ramo.strategy.rst>
   Utility Functions <ramo.utility_function.rst>
   Printing <ramo.printing.rst>
   Other <ramo.utils.rst>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
