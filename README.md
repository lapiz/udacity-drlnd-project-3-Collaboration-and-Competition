# Project 3: Collaboration and Competitation

This repository contains an implementation of project 3 for [Udacity's Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Project Details

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

My agents get average (over 100 episodes) of those scores (maximun value of two agents) is at least +0.5.

## Getting Started

### Python environment

- If you run this project on your own environment, you install some packages.
  - Python == 3.6
  - pytorch == 0.4
  - mlagents == 0.4 (Unity ML Agents)
- Or you can run this project on jupyter notebook.

### Dependencies

To set up your python environment (with conda) to run code in the project, follow the intstruction below.

- Create and activate a new envirionment with Python 3.6

```bash
conda create --name project3 python=3.6
conda activate project3
```

- Clone my project repository and install requirements.txt

```bash
git clone https://github.com/lapiz/udacity-drlnd-project-3-Collaboration-and-Competition.git
cd udacity-drlnd-project-3-Collaboration-and-Competition
pip install -r requirements.txt
```

### Downloading the Unity environment

Different versions of the Unity environment are required on different operational systems.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## Instructions

- Command lines
  - If train yourself, run it. you can skip hparam filename, it found 'default.json' file.

```bash
python train.py hparam.json
```

- Notebook
  - If train, open [Tennis.ipynb](Tennis.ipynb)
  - If run with already trained data, open [run_trained.ipynb](run_trained.ipynb)

## Files

- README.md
  - This file
- requirements.txt
  - python environment requirements packages
  - Use pip with -r options
- Tennis.ipynb
  - Main notebook file.
  - Based on udacity project skelecton notebook
  - I implemented my agent and some helper classes.
- run_trained.ipynb
  - Run with trained data sample
- Report.ipynb
  - My Project report.
  - Include these things
    - Learning Algorithm
      - Hyperpameters
      - Model architechures
    - Plot of Rewards
- default_agent_0_actor.pth
  - trained model weights for actor of agent 0
- default_agent_0_critic.pth
  - trained model weights for critic of agent 0
- default_agent_1_actor.pth
  - trained model weights for actor of agent 1
- default_agent_1_critic.pth
  - trained model weights for critic of agent 1
- scores.py
  - helper code for score data
- train.py
  - Train ddpg agents.
  - Based on udacity DRLND ddqg-bipdel sample project and my second project
  - Run command line with hparam json arguments for test
- ddpg_agent.py
  - DDPG Agent implementation with PlayBuffer
  - Based on udacity DRLND ddqg-bipdel sample project and my second project
  - Remove shared network from my second projects
- model.py
  - Model described by hidden layers (Actor and Critic)
  - same as my second project