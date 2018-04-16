import gym
import numpy as np

# Setup Enviornment
env = gym.make('CartPole-v0')

bestparams = None
bestreward = 0

def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        env.render()
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


# Initial weights
parameters = np.random.rand(4) * 2 - 1
noiseLevel = 1.
for i in range(10000):
    changeParameters = parameters+(np.random.rand(4) * 2 - 1)*noiseLevel
    reward = run_episode(env,changeParameters)
    if reward > bestreward:
        print(reward)
        bestreward = reward
        parameters = changeParameters
        # considered solved if the agent lasts 200 timesteps
        if reward == 200:
            break
