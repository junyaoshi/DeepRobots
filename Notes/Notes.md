# DeepRobots Research Meeting Notes

---
## To-Do List

### Discrete Implementation Details

- state space (done)
    - self.state
    - (theta, a1, a2), where theta is (-pi, pi) with pi/32 interval, a1 is (0, pi/8) with pi/32 interval, a2 is (-pi/8, 0) with pi/32 interval
- action space
    - joint velocities (adot1, adot2), range = (-pi/8, pi/8) not including the case where (adot1 = 0, adot2 = 0), intervals = 0.01
    - when choosing a random action, in a while loop, check if the action would result in valid a1 and a2 values through integration, only break when it is valid
- reward
    - negative reward for singularity (a1 = a2) = -(link_length)*10
    - negative reward based on proximity of a1 and a2 (using log function, log 0 = -inf)
    - body_v[0] / (a1dot^2 + a2dot^2) * some constant for scaling
        - displacement from current state to next state along x axis is x, x dot is velocity
- transition model (done)
- goal state (no goal state)

## Timeline

1. dicrete RL implementation
2. Deep RL
3. continuous implementation (python function sovler to solve ODE)
4. DDPG

## 2/1

- questions:
    - how to resolve weird state parameters in RL?
- To-do
   - change state space to (theta, a1, a2), where theta is (-pi, pi) with pi/32 interval, a1 is (0, pi/8) with pi/32 interval, a2 is (-pi/8, 0) with pi/32 interval
   - when theta goes out of range, in move(), add or minus 2pi depending on whether it is positive or negative
   - when choosing a random action, in a while loop, check if the action would result in valid a1 and a2 values through integration, only break when it is valid
   - implement graphing of x-velocity, x displacement, joint angles etc. in policy testing


## 1/25

- questions:
    - how to resolve weird state parameters in RL?
- To-do
   - change state space to (theta, a1, a2), where theta is (-pi, pi) with pi/32 interval, a1 is (0, pi/8) with pi/32 interval, a2 is (-pi/8, 0) with pi/32 interval
   - when theta goes out of range, in move(), add or minus 2pi depending on whether it is positive or negative
   - when choosing a random action, in a while loop, check if the action would result in valid a1 and a2 values through integration, only break when it is valid
   - implement graphing of x-velocity, x displacement, joint angles etc. in policy testing


## 12/27

- questions:
    - is my model working correctly?
- observation:
    - time_interval is dependent on angle_interval, if t_interval is too small, and change in angle_interval is too small, there might not be any update
    - afraid that this might be a problem?
- Deep RL:
    - loss function = expected reward - groundtruth reward
- RL implementation:
    - always use epsilon-greedy approach
     - epsilon should be something that decreases over time
    - (optional)in general, joint velocities are sinusoidal -> this is how we should let the robot choose 
       - Tony will think about this more


    
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
