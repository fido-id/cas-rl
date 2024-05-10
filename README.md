

<p align="center">
  <img src="https://github.com/fido-id/cas-rl/assets/19633559/1bbf3fff-8096-4843-bdd1-cf79f3e2f895" alt="Logo" width="1000">
</p>
<h1 align="center">Collision Avoidance System PoC</h1>
<p align="center">
  <img alt="CI" src="https://github.com/fido-id/cas-rl/actions/workflows/code-quality-checks.yaml/badge.svg">
  <img alt"Python version" src="https://img.shields.io/badge/python-3.11-blue">
  <img alt"Coverage" src="https://raw.githubusercontent.com/fido-id/cas-rl/gh-pages/coverage.svg">
</p>


## 💡 Overview

This repository contains a Proof of Concept (PoC) implementation of a Collision Avoidance System implemented in Python using [Pygame](https://www.pygame.org/) for graphics rendering, and backed with reinforcement learning using [Q-Learning](https://en.wikipedia.org/wiki/Q-learning) algorithm.

## Table of Contents

- [Preview](#-preview)
- [Implementation details](#-implementation-details)
- [Requirements](#-requirements)
- [Installation](#%EF%B8%8F-installation)
- [Usage](#-usage)


## 🚀 Preview

![game](https://github.com/fido-id/cas-rl/assets/19633559/a1986275-a6bf-45f4-b0ec-db426884b04f)

## 🔎 Implementation details

This project utilizes Q-Learning as the learning algorithm for the Collision Avoidance System (CAS). The Q-Learning algorithm is a reinforcement learning technique that enables the CAS to learn optimal actions to avoid collisions based on previous experiences.

### Training Phase
During the initial phase, termed as the training phase, the CAS learns to navigate the environment without user interaction. Upon execution, a Pygame window spawns, and the CAS undergoes training for a specified number of episodes. Throughout this phase, the CAS explores the environment and updates its action-value function using the [Q-Learning update rule](https://en.wikipedia.org/wiki/Q-learning#Algorithm). The action-value function represents the expected utility of taking a particular action in a given state, guiding the CAS towards making informed decisions to avoid collisions. During this phase, to optimize process, the graphics is not loaded. Moreover, it is possible to control the FPS using the keys Q and E of the keyboard. Pressing E increases the FPS making the training process faster, pressing Q decreases the FPS slowing down the training.

### Gameplay
Once the training phase concludes, users can engage in gameplay. The CAS, having learned from its training, now applies its acquired knowledge to navigate the environment and avoid collisions effectively. Users can interact with the CAS within the Pygame window, observing its intelligent decision-making process as it maneuvers through the environment. To control the spaceship the user can use the WASD set of keys.

### Simplifications
During project development, simplifications were made to focus on Q-learning:

 - Episode Termination: When agents stray out of bounds, episodes end and new ones begin, ensuring training remains within the defined environment.
 - Action Space: Limited to five dimensions (STAY, UP, DOWN, LEFT, and RIGHT), streamlining decision-making.
 - Constant Speed: Entities maintain a fixed speed, removing complexities of acceleration and deceleration.

## 🪐 Requirements
To run the code successfully, ensure you have the following prerequisites:
 - Python 3.11 or higher.
 - [Poetry](https://python-poetry.org/) installed for managing dependencies and packaging efficiently.

## 🛠️ Installation
 1. Clone the repository
```bash
git clone https://github.com/fido-id/cas-rl.git
```
 2. Install the dependencies
```bash
cd cas-rl
poetry install
```

## 🛸 Usage 

There are two main scripts: `train.py` and `play.py`. Before running the script to play, a first training phase is needed.

To start the training phase run:

```bash
poetry run train
```

There are some parameters that can be set, they are all listed in the `--help` of the script:
```
usage: Collision Avoidance System PoC [-h] [-sp STATE_SAVE_PATH] [-lp STATE_LOAD_PATH] [-nu N_UFOS] [-e EPISODES]

This script is used to train a number of qlearning agent to avoid collisions from another agent.
The agent trying to escape will be UFOs, while the agent trying to collide with them will be aspaceship.
The training will be done with an adversarial setting.

options:
  -h, --help            show this help message and exit
  -sp STATE_SAVE_PATH, --save-path-to STATE_SAVE_PATH
                        Path to the the folder containing the state of the agent. If not given, it will default to the current time in YYYYMMDD-HHMM format
  -lp STATE_LOAD_PATH, --load-pretrained-from STATE_LOAD_PATH
                        Path to the the folder containing the pretrained state of the agent. This is useful whenwe want to start train from an already trained episode
  -nu N_UFOS, --n-ufos N_UFOS
                        Number of ufos to spawn
  -e EPISODES, --episodes EPISODES
                        Number of episodes to train for
```

Once the `train.py` has finished with the execution, the script `play.py` can be run.

To start the gameplay, run:

```bash
poetry run play
```
In this case, the only parameter that can be passed to the script is the path to the trained state in the previous step.

```
This script is used to play with a trained agent. The latter must avoid collision from a player that can freely move inside a predefined grid. When the UFO goes out of bound the game restarts

options:
  -h, --help            show this help message and exit
  -lp STATE_LOAD_PATH, --load-pretrained-from STATE_LOAD_PATH
                        Path to the the folder containing the pretrained state of the agent.
```
