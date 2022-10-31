Algorithms
=====================
Ramo currently contains more than 10 different algorithms capable of computing or learning (approximate) equilibria in multi-objective games. The following tables give an overview of our current algorithms, the equilibria they aim to find and the utility functions they support.

Computing Equilibria
----------------------
The following table gives an overview of the current algorithms for computing equilibria from a given MONFG and utility functions.

+--------------------------------------+-------------------------------+-------------------+
| Algorithm                            | Equilibrium                   | Utility functions |
+======================================+===============================+===================+
| MOSE                                 | Pure-strategy Nash equilibria | All               |
+--------------------------------------+-------------------------------+-------------------+
| MOQUPS                               | Pure-strategy Nash equilibria | Quasiconvex       |
+--------------------------------------+-------------------------------+-------------------+


Learning Algorithms
---------------------
This table show the learning algorithms currently included in Ramo. Please note that only the iterated best-response and fictitious play algorithms get access to their full individual payoff matrices. All other algorithms have to learn the payoffs together with an optimal strategy.

+--------------------------------------+-------------------------------+-------------------+
| Algorithm                            | Equilibrium                   | Utility functions |
+======================================+===============================+===================+
| Iterated best-response               | Nash equilibrium              | All               |
+--------------------------------------+-------------------------------+-------------------+
| Fictitious play                      | Nash equilibrium              | All               |
+--------------------------------------+-------------------------------+-------------------+
| Independent Q-learners               | Nash equilibrium              | All               |
+--------------------------------------+-------------------------------+-------------------+
| Independent actor-critic learners    | Nash equilibrium              | All               |
+--------------------------------------+-------------------------------+-------------------+
| Joint-action Q-learners              | Nash equilibrium              | All               |
+--------------------------------------+-------------------------------+-------------------+
| Joint-action actor-critic learners   | Nash equilibrium              | All               |
+--------------------------------------+-------------------------------+-------------------+
| Cooperative action communication     | Nash equilibrium              | All               |
+--------------------------------------+-------------------------------+-------------------+
| Self-interested action communication | Nash equilibrium              | All               |
+--------------------------------------+-------------------------------+-------------------+
| Cooperative policy communication     | Nash equilibrium              | All               |
+--------------------------------------+-------------------------------+-------------------+
| Hierarchical communication           | Nash equilibrium              | All               |
+--------------------------------------+-------------------------------+-------------------+
| Non-stationary agent                 | Leadership equilibrium        | All               |
+--------------------------------------+-------------------------------+-------------------+
| Best-response agent                  | Leadership equilibrium        | All               |
+--------------------------------------+-------------------------------+-------------------+
