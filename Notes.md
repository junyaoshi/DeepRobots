# DeepRobots Research Meeting Notes

---
## To-Do List

## Discrete Implementation Details

- state space (done)
    - self.state
- action space
    - joint velocities (adot1, adot2), range = (-1, 1), intervals = 0.01
- reward
    - negative reward for singularity (a1 = a2) = -(link_length)*10
    - x_dot / (adot1^2 + adot2^2)
        - displacement from current state to next state along x axis is x, x dot is velocity
- transition model (done)
- goal state (no goal state)

## Timeline

1. dicrete RL implementation
2. Deep RL
3. continuous implementation (python function sovler to solve ODE)
4. DDPG

## 12/27

- questions:
    - is my model working correctly?
- observation:
    - time_interval is dependent on angle_interval, if t_interval is too small, and change in angle_interval is too small, there might not be any update
    - afraid that this might be a problem?
- Deep RL:
    - loss function = expected reward - groundtruth reward

    
## 12/6

- next time
    - look at Atari paper for Deep RL implementation details
- questions:
    - what should self.state contain?
    - implement discretized angles?
    - are mutator/accessor methods useless?
    - Deep RL link

## 11/30

- next time
    - fix DeepRobots implementation
    - ask Tony for Mathematica inputs and outputs
    - compare with Python model
- questions:
    - openAI (don't worry)
    - how to keep the continuity Scott mentioned (don't worry)
    - this implementation is only for proximal link, do we need other links? (implement the function)
    - do I need to integrate over theta_dot? Is it already given by x and y dot? (No)
    - when the input is pi/3, pi/3, D becomes 0 (singularity)
    - self.state is broken (another version of implementation) 
    - plot graph with matplotlib (noted)
