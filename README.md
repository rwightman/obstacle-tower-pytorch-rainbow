# Obstacle Tower Challenge - Rainbow & PyTorch

This repo contains my starting point for Round 1 of the Obstacle Tower Challenge -- a clone of the [starters kit](https://github.com/Unity-Technologies/obstacle-tower-challenge) with Kaixhin's great implementation of Rainbow as my baseline RL algorithm (https://github.com/Kaixhin/Rainbow).

I hacked the rainbow impl to work with color images, unstacked frames. Created an env wrapper for allowing reward shaping, custom resolutions, and other changes.

The end resulted worked better than expected, notably better than the recommended Dopamine RAINBOW as a baseline. With a bit of tuning, an agent can be trained to average floor 7-8 over 3-4 days of training. The agent can hit floor 10 fairly often but has troubles moving past that point. There is high variability in the performance though.

Moving forward, I was working on a PyTorch impl of R2D2 with the addition of novelty/curiosity to the reward, so Rainbow-like with an RNN policy and distributed training. I may publish that at some point if I move it forward and apply it successfully to a future task.

## Usage
* Setup a new Conda Python 3.6 environment (do not use 3.7! compatibility issues with Unity's support modules)
* Install recent (ver 1.x) of PyTorch
* Setup environment download engine as per: https://github.com/Unity-Technologies/obstacle-tower-challenge#local-setup-for-training but using this repo in place of that clone and do it within the same Conda env
* Create a folder named 'results' to put in model and evaluation report
* Run train_obt.py and wait...
* run.py can be used to run the trained models for submission or for viewing with the `--realtime` flag set
