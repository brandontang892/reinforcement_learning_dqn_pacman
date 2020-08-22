# convolutional_dqn_pacman
Convolutional Neural Networks (CNN) with Deep Q-Learning (DQN) Agent for OpenAI Pacman Domain

Implementation utilizes a target network to guide learning in the right direction and takes advantage of experience replay to prevent state transition dependencies from interfering with learning. The Markov Decision Process and overall environment are defined/provided by OpenAI. Tensorboard was integrated into this project for training/progress visualizations. States are represented by three dimensional Red-Green-Blue (RGB) arrays which make convolutional neural networks (CNN) well-suited for the task of training the agent. 

**Notes:** 
- Empirically, running the DQN model with multiple passes (saving weights from previous pass and running model again initialized with those weights) leads to better performance because the exploration/exploitation epsilon constant is allowed to re-decay, effectively helping the agent escape from local "traps" and not get stuck during training. Essentially, the agent gets to pick up from where it ended in the last pass, except with a fresh pair of eyes.
- In contrast to other discrete grid world domains like the Taxi domain, the Pacman environment is represented as an 210 x 160 x 3 array. The Taxi environment is only a 5 x 5 x 3 array in comparison. This means our state space is much larger, which would lead to infeasibly long training times for traditional tabular reinforcement algorithms like Q-learning. This is why Deep Q-Learning (DQN) is a sensible choice for this environment; still, each episode can take several minutes to train.


## Prerequisites

- Create new environment in Anaconda:  
```
conda create --name reinforcement_learning
conda activate reinforcement_learning
```
Install the following in your newly created Anaconda environment:
- Tensorflow Version 1.15
```
pip install tensorflow==1.15
```
- OpenAI Gym
```
pip install gym
conda install git
pip install git+https://github.com/Kojoley/atari-py.git
pip install gym[all]
```
- NumPy
```
pip install numpy
```
- tqdm
```
pip install tqdm
```

## Running the DQN Agent

To train the agent, cd into the root project directory and type "python DQNAgent_Pacman.py" in the command terminal. The model will automatically save its weights when a certain threshold of performance is reached.

To render/display the agent interacting with the environment, set **SHOW_PREVIEW** to **True** and set **AGGREGATE_STATS_EVERY** to **1** to see rendering for every timestep. 

To see training visualizations on Tensorboard, you should type something like **tensorboard --logdir=pacman-1597569936/** in the terminal. pacman-1597569936 should be replaced by the name of the folder in the log folder.

Then, just copy the link that the terminal gives you into your browser. If this link doesn't work, try searching http://localhost:6006/ 


## OpenAI Pacman Domain:

<img width="175" height="175" src="https://gym.openai.com/videos/2019-10-21--mqt8Qj1mwo/MsPacman-v0/poster.jpg">  

Agent actions, agent/ghost interactions all follow the traditional rules of the Pacman game.

