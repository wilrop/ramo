# Communication in Multi-Objective Normal-Form Games
This repo consists of five different types of agents that we have used in our study of communication in multi-objective normal-form games. The settings that involve communication do this following a leader-follower model as seen in Stackelberg games. 
In such settings, agents switch in a round-robin fashion between being the leader and communicating something and being the follower and observing the communication.

## No communication setting
In this setting two agents play a normal-form game for a certain amount of episodes. This experiment serves as a baseline for all other experiments.

## Cooperative action communication setting
In this setting, agents communicate the next action that they will play. The follower uses this message to pre-update their policy. This setting is similar to Iterated Best Response and attempts to find the optimal joint policy.

## Competitive action communication setting
This setting places the agents in a more competitive environment. This means that agents learn a specific best-response policy to every possible message. As such, agent's are not optimising for an optimal joint policy, but rather are acting in a self-interested manner.

## Cooperative policy communication setting
This setting follows the same dynamics as the cooperative action communication setting, but communicates the entire policy instead of the next action that will be played.

## Optional communication setting
The last setting gives agents the chance to learn for themselves whether communication helps them. All agents learn a top-level policy that chooses whether they will communicate when they are the leader or not. They also have two low-level agents, one "no communication agent" and one agent that does communicate. Which agent that is used as the communicating agent, is completely optional. When agents choose to communicate, they utilise their lower level communicating agent. When agents opt out of communication, they utilise their lower level no communication agent.

## Getting Started

Experiments can be run from the `MONFG.py` file. There are 5 MONFGs available, having different equilibria properties under the SER optimisation criterion, using the specified non linear utility functions. You can also specify the type of experiment to run and other parameters. 

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details


