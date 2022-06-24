.. Ramo documentation master file, created by
   sphinx-quickstart on Fri Mar 11 16:06:15 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Ramo's documentation!
================================

The main focus of this repository is to develop and maintain different algorithms for learning or computing equilibria in Multi-Objective Normal-Form Games (MONFGs).

Below we provide a handy table giving an overview of our current algorithms, their type and the equilibrium concept they aim to find.

+--------------------------------------+---------+-------------------------------+
| Algorithm                            | Type    | Equilibrium                   |
+======================================+=========+===============================+
| MOQUPS                               | Compute | Pure-strategy Nash equilibria |
+--------------------------------------+---------+-------------------------------+
| Iterated best-response               | Learn   | Nash equilibrium              |
+--------------------------------------+---------+-------------------------------+
| Fictitious play                      | Learn   | Nash equilibrium              |
+--------------------------------------+---------+-------------------------------+
| Independent Q-learners               | Learn   | Nash equilibrium              |
+--------------------------------------+---------+-------------------------------+
| Independent actor-critic learners    | Learn   | Nash equilibrium              |
+--------------------------------------+---------+-------------------------------+
| Joint-action Q-learners              | Learn   | Nash equilibrium              |
+--------------------------------------+---------+-------------------------------+
| Joint-action actor-critic learners   | Learn   | Nash equilibrium              |
+--------------------------------------+---------+-------------------------------+
| Cooperative action communication     | Learn   | Nash equilibrium              |
+--------------------------------------+---------+-------------------------------+
| Self-interested action communication | Learn   | Nash equilibrium              |
+--------------------------------------+---------+-------------------------------+
| Cooperative policy communication     | Learn   | Nash equilibrium              |
+--------------------------------------+---------+-------------------------------+
| Hierarchical communication           | Learn   | Nash equilibrium              |
+--------------------------------------+---------+-------------------------------+
| Non-stationary agent                 | Learn   | Leadership equilibrium        |
+--------------------------------------+---------+-------------------------------+
| Best-response agent                  | Learn   | Leadership equilibrium        |
+--------------------------------------+---------+-------------------------------+

We also provide a number of handy utilities. For example, this repository comes with a number of known and pre-analysed example games and utility functions. There are also helper functions to generate random games, scalarise games, etc. In addition, there is functionality to save and plot data resulting from experiments
and a bunch more, with even more on the way!

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   Installation
   Getting started
   Examples
   Citation

.. toctree::
   :maxdepth: 1
   :caption: Algorithms

   ramo.best_response

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
