# DeepRobots
Deep Reinforcement Learning for Snake Robot Locomotion

---
## Contributors
- Jack Shi (junyaoshi)
- Tony Dear (tonydear)
- Jesse Redford (Jesse-Redford)
- Jose Ronaldo Pinheiro Carneiro Filho (JRPCF)
- Jasmine Wang (yw2946)

## Dependencies
- Python ≥ 3.5
- keras ≥ 2.2.4 (For own DQN implementation only)
- matplotlib 
- numpy
- numba
- scipy
- stable-baselines
- PyBullet


## Installation

Create a conda virtual environment
```bash
conda create -n DeepRobots python=3.7 numpy scipy matplotlib numba 
conda activate DeepRobots
``` 

Install PyBullet with pip
```bash
[path-to-conda-pip] install pybullet
```
where `[path-to-conda-pip]` is the path to pip in the conda virtual environment. For example, on Mac OS, it can be `~/opt/anaconda3/envs/DeepRobots/bin/pip`, and on Windows, it can be `[path-to-anaconda3]/envs/DeepRobots/Scripts/pip`. An example of the full pip install command in Mac OS Terminal is:
```bash
~/opt/anaconda3/envs/DeepRobots/pip install pybullet
```

The purpose of using the conda pip is to only install package in the conda virtual environment instead of installing globally

Install stable-baselines by following [this guide](https://stable-baselines.readthedocs.io/en/master/guide/install.html). Make sure to install the DEVELOPMENT VERSION instead of the stable release or the bleeding-edge version. Make sure the use conda's pip for any pip installation commands. Remove the cloned stable-baselines after the installation, because there is already a custom version of stable-baselines in this repo. 

## Physical Robot Setup

Follow [this guide](docs/Deeprobots_setup_instructions_rasberrypi.txt) to set up the physical robot using Rasperry Pi.

## Directories
- `discrete_rl`: code for training RL agent in discrete state and action spaces; no longer maintained
- `docs`: documentation files 
- `DQN`: our implementation of DQN (we are in the process of switching to stable-baselines instead of using this)
    - `DQN_agent.py`: a general-purpose class that can be easily used to train different kinds of snake robots. 
    - `DQN_runner.py`: a simple interface for running a single trial of DQN with customized parameters
    - `DQN_multi_runner.py`: a simple interface for running a multiple trials of DQN with customized parameters
- `DQN_old`: some old implementation of DQN algorithm and runners
- `envs`: high-level OpenAI Gym environments that can directly be used by stable-baselines algorithms in training
- `Jesses_Folder`: scripts used by Jesse Redford
- `Mathematica Notebooks`: Mathematica notebooks used for generating animation, plots, and deriving mathematical models
- `Notes`: old meeting notes; no longer used
- `OpenAiGym_Old`: old code for OpenAI Gym and baselines integration; no longer used or maintained
- `Robots`: different types of low-level snake robot models; need be wrapped in OpenAI Gym environment format to be trained by stable-baselines
- `stable-baselines`: RL algorithms implemented by stable-baselines
- `training_scripts`: various stable-baselines training/evaluation scripts
- `utils`: various utility scripts

## Quick Start Guide

```bash
conda activate DeepRobots
```

Then navigate to DeepRobots/training_scripts, and run a script of your choice.

## Tips

Take a look at [stable-baselines documentation](https://stable-baselines.readthedocs.io/en/master/), especially the "Getting Started" and "Tensorboard Integration" sections.

## Previous Documentation

Here is a list of previous documentation that is no longer maintained:
- [Usage of our own version of DQN algorithm](docs/DQN_old.md) for training simulated and physical robots




