# Ramo
A collection of algorithms and utilities for computing or learning (approximate) equilibria in multi-objective games.


## Features
The main focus of this repository is to develop and maintain different algorithms for learning or computing equilibria in Multi-Objective Normal-Form Games (MONFGs). Below we provide a handy table giving an overview of our current algorithms, their type and the equilibrium concept they aim to find.

| Algorithm                            | Type  | Equilibrium                   |
|--------------------------------------|-------|-------------------------------|
| MOQUPS                               | Compute | Pure-strategy Nash equilibria |
| Iterated best-response               | Learn | Nash equilibrium              |
| Fictitious play                      | Learn | Nash equilibrium              |
| Independent Q-learners               | Learn | Nash equilibrium              |
| Independent actor-critic learners    | Learn | Nash equilibrium              |
| Joint-action Q-learners              | Learn | Nash equilibrium              |
| Joint-action actor-critic learners   | Learn | Nash equilibrium              |
| Cooperative action communication     | Learn | Nash equilibrium              |
| Self-interested action communication | Learn | Nash equilibrium              |
| Cooperative policy communication     | Learn | Nash equilibrium              |
| Hierarchical communication           | Learn | Nash equilibrium              |
| Non-stationary agent                 | Learn | Leadership equilibrium        |
| Best-response agent                  | Learn | Leadership equilibrium        |

We also provide a number of handy utilities. For example, this repository comes with a number of known and pre-analysed example games and utility functions. There are also helper functions to generate random games, scalarise games, etc. In addition, there is functionality to save and plot data resulting from experiments
and a bunch more, with even more on the way!

## Contributing
We are attempting to build a library containing the state of the art in multi-objective game theory. If you are working on similar stuff and want to get it included, contributions are very welcome! If you are considering this, please send me a message (willem.ropke@vub.be) so we can discuss how to move forward.

## Citation
To cite the usage of this repository please use the following:
```
@misc{ropke2022ramo,
  author = {Willem Röpke},
  title = {Ramo: Rational Agents with Multiple Objectives},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/wilrop/mo-game-theory}},
}
```
This repository contains (derivatives of) original work that has been published in the literature. We present a complete overview of which algorithms come from which work in [references.md](references.md).

## License
This project is licensed under the terms of the MIT license.
