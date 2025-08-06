# Reinforcement Learning

This repository contains code and other resources used in the study of reinforcement learning following the second edition of the book **"Reinforcement Learning: An Introduction"** by Richard S. Sutton and Andrew G. Barto.

The code in each folder can be run independently.

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/BibarelUsedFly/reinforcement-learning
cd reinforcement-learning
```

Install dependencies

```bash
pip install -r requirements.txt
```

## Contents 
### N-Armed Bandits
Bandits.py replicates the 10-armed testbed used in the book to compare performance of ε-greedy algorithms with varying exploration rate ε.
Running Bandits.py will show some curves representing the competing agents.

dicegame.py will show an interactive game where a simple agent has to roll a die each timestep. The possible actions are a d4, d6, d8, d10, d12 and d20. While the obvious best action is to roll a d20, the agent has no way to know this, and chance may have it that a better action yields worse results. The display shows the Q-Table evolution of the agent as it learns from actions.
Game parameters can be adjusted in parameters.py

### Dynamic Programming
This section implements three algorithms to obtain the best policy for navigating a simple gridworld. The differently colored square marks the goal.
The default reward is -1.0 for each step. The actions are ("↑", "↓", "→", "←").
The starting policy is a random policy with probability 0.25 for each action.

The interface shows the grid with the estimated value of each state under the current policy.

The arrow keys can be used to navigate the simulation. Left and right arrows navigate a timestep (each timestep is a policy evaluation). Up and down arrows navigate policies, which are improved when policy evaluation is done. The simulation is done when the new policy is identical to the previous policy; then we know we've reached an optimal policy.

The "P" key can be used to alterate between showing the current state values and the current policy. Policies are shown as arrows where the arrow size is proportional to the probability o choosing an action. A deterministic policy will have a single big arrow in each square, whereas a random policy will have four little arrows pointing in different directions.

PolicyIteration.py runs policy evaluation until the state values' change is less than a theta parameter. Then it improves the policy to a greedy one considering the new state value estimates.

ValueIteration.py runs policy evaluation once and immediately improves the policy to a greedy one considering the new state value estimates.

ValueIterationPizzaWalls.py intoduces walls that block agent movement and a pizza with a (by default) positive reward to add some complexity. In this scenario, the purple-ish numbers and arrows represent the states where the pizza is not yet consumed, whereas the black arrows represent the grim reality where there is no more pizza.

Simulation parameters can be adjusted in parameters.py

### Model Free Learning
This section contains an environment (defined in Models.py) where a little robot tries to reach a goal in a gridworld, but the goal is protected by the asylum demon from Dark Souls. Reaching the goal yields a positive reward, but colliding with the monster means death (and thus, a negative reward).

Some model-free tabular learning algorithms are implemented in Agents.py, and Simulate.py can be ran to show a simple simulation where the robot uses the knowledge stored in "Checkpoint3000.npy" (the result of training for 3000 episodes) to try and reach the goal. The simulation can be interacted with using the arrow keys; right moves time forward and left moves it backwards.

### Montecarlo
Montecarlo.py shows a simple 3x3 gridworld environment alongside a Q-Table to illustrate how rewards propagate using the montecarlo algorithm. The right side of the iterface shows a 3x3 grid with four values on each cell representing the estimated values (Q-table entries) of each action possible in that cell.

First, you can use the arrow keys to move little Bidoof and decide on a trajectory. Each step has a reward of -1, but touching the skull has a reward of -10. Upon reaching the goal, you can use the right arrow key to update the Q-table one step at a time, propagating the reward backwards from the end of the episode, using the real reward obtained by the trajectory according to the Montecarlo model-free algorithm. The larning rate (ALPHA) and the discount factor (GAMMA) can also be adjusted.