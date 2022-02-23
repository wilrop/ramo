# Multi-Objective Game Theory
A collection of algorithms and utilities for calculating or learning (approximate) equilibria in multi-objective games.


## Features
The main focus of this repository is to develop and maintain different algorithms for learning or calculating equilibria in Multi-Objective Normal-Form Games (MONFGs). Below we provide a handy table giving an overview of our current algorithms, their type and the equilibrium concept they aim to find.

| Algorithm                            | Type        | Equilibrium                   |
|--------------------------------------|-------------|-------------------------------|
| PSNE                                 | Calculating | Pure-strategy Nash equilibria |
| Iterated best-response               | Learning    | Nash equilibrium              |
| Fictitious play                      | Learning    | Nash equilibrium              |
| Independent Q-learners               | Learning    | Nash equilibrium              |
| Independent actor-critic learners    | Learning    | Nash equilibrium              |
| Joint-action Q-learners              | Learning    | Nash equilibrium              |
| Joint-action actor-critic learners   | Learning    | Nash equilibrium              |
| Cooperative action communication     | Learning    | Nash equilibrium              |
| Self-interested action communication | Learning    | Nash equilibrium              |
| Cooperative policy communication     | Learning    | Nash equilibrium              |
| Hierarchical communication           | Learning    | Nash equilibrium              |
| Non-stationary agent                 | Learning    | Leadership equilibrium        |
| Best-response agent                  | Learning    | Leadership equilibrium        |

We also provide a number of handy utilities. For example, this repository comes with a number of known and pre-analysed example games and utility functions. There are also helper functions to generate random games, scalarise games, etc. In addition, there is functionality to save and plot data resulting from experiments
and a bunch more, with even more on the way!

## Contributing
We are attempting to build a library containing the state of the art in multi-objective game theory. If you are working on similar stuff and want to get it included, contributions are very welcome! If you are considering this, please send me a message (willem.ropke@vub.be) so we can discuss how to move forward.

## References
This repository contains (derivatives of) original work that has been published in several papers. We present here a link to the original resources and citations.
### Preference Communication in Multi-Objective Normal-Form Games
This work developed several communication protocols.

[paper](https://arxiv.org/abs/2111.09191) | [repo](https://github.com/wilrop/communication_monfg)

```
@misc{ropke2021preference,
      title={Preference Communication in Multi-Objective Normal-Form Games}, 
      author={Willem Röpke and Diederik M. Roijers and Ann Nowé and Roxana Rădulescu},
      year={2021},
      eprint={2111.09191},
      archivePrefix={arXiv},
      primaryClass={cs.GT}
}
```

### On Nash Equilibria in Normal-Form Games With Vectorial Payoffs
This work developed the *PSNE* algorithm for finding all pure-strategy Nash equilibria in games with quasiconvex utility functions.

[paper](https://arxiv.org/abs/2112.06500) | [repo](https://github.com/wilrop/Nash-Equilibria-MONFG)

```
@misc{ropke2021nash,
      title={On Nash Equilibria in Normal-Form Games With Vectorial Payoffs}, 
      author={Willem Röpke and Diederik M. Roijers and Ann Nowé and Roxana Rădulescu},
      year={2021},
      eprint={2112.06500},
      archivePrefix={arXiv},
      primaryClass={cs.GT}
}
```

### Commitment and Cyclic Strategies in Multi-Objective Games
This work developed the non-stationary commitment and best-response learners.

[repo](https://github.com/wilrop/Cyclic-Equilibria-MONFG)

*No citation available yet.*

### A utility-based analysis of equilibria in multi-objective normal-form games
The first work to introduce independent multi-objective Q-learners.

[paper](https://www.cambridge.org/core/journals/knowledge-engineering-review/article/abs/utilitybased-analysis-of-equilibria-in-multiobjective-normalform-games/00229BD20BB55C72B7DF80007E5725E1) | [arXiv](https://arxiv.org/abs/2001.08177) | [repo](https://github.com/rradules/equilibria_monfg)

```
@article{radulescu2020utility,
    author = {Rădulescu, Roxana and Mannion, Patrick and Zhang, Yijie and Roijers, Diederik M. and Now{\'{e}}, Ann},
    doi = {10.1017/S0269888920000351},
    journal = {The Knowledge Engineering Review},
    pages = {e32},
    publisher = {Cambridge University Press},
    title = {{A utility-based analysis of equilibria in multi-objective normal-form games}},
    volume = {35},
    year = {2020}
}
```

### Opponent Modelling for Reinforcement Learning in Multi-Objective Normal Form Games
The first work to introduce independent multi-objective actor-critic learners.

[paper](https://www.ifaamas.org/Proceedings/aamas2020/pdfs/p2080.pdf) | [full paper](https://ala2020.vub.ac.be/papers/ALA2020_paper_32.pdf)

```
@inproceedings{zhang2020opponent,
    address = {Auckland, New Zealand},
    author = {Zhang, Yijie and Rădulescu, Roxana and Mannion, Patrick and Roijers, Diederik M. and Now{\'{e}}, Ann},
    booktitle = {Proceedings of the 19th International Conference on Autonomous Agents and MultiAgent Systems},
    isbn = {9781450375184},
    pages = {2080--2082},
    publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
    series = {AAMAS '20},
    title = {{Opponent Modelling for Reinforcement Learning in Multi-Objective Normal Form Games}},
    year = {2020}
}
```

## License
This project is licensed under the terms of the MIT license.
