# DeepRobots
Deep Reinforcement Learning for Robot Locomotion

---
## Contributors
- Jack Shi (junyaoshi)
- Tony Dear (tonydear)

## Dependencies
- Python ≥ 3.6
- Keras 2.2.4
- matplotlib 3.1.0

If python requirement is not met, a virtual environment can be installed 
using the follwoing method on terminal:

```bash
conda create -n py37 python=3.7 anaconda 
conda activate py37
pip install keras=2.2.4
pip install matplotlib=3.1.0
``` 

## Directories
- `DiscreteRL`: different versions of implementation of discrete RL agent
- `DQN`: most up-to-date implementation of DQN agent
    - `DQN_agent.py`: a general-purpose class that can be easily used to train different kinds of snake robots. 
    See `DQN_swimming_runner.py` and `DQN_wheeled_runner.py` for examples
- `DQN_old`: some old implementation of DQN, please ignore
- `Mathematica Notebooks`: used for generating animation and plots
- `Robots`: different types of snake robot models.
    - `DiscreteDeepRobots.py`, `ContinuousDeepRobot.py`: wheeled snake robot implementation
    - `ContinuousSwimmingBot.py`: swimming snake robot implementation

## Quick Start Guide

Clone this repository into some local directory ```local_dir``` on the local machine.

First, open terminal and change directory: 

```cd local_dir``` 

Clone this repository in the local directory: 

```git clone https://github.com/junyaoshi/DeepRobots.git```

Then, change directory to DeepRobots directory:

```cd DeepRobots```

Make sure the system path is set up properly so that all library and module imports can be successfully performed. 
See `DQN_swimmimming_runner.py` for example.

```bash
python DQN/DQN_swimming_runner.py
```
