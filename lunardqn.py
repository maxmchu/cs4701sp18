import gym
import numpy as np
from dqnClass import DQN

# Setup Enviornment
env = gym.make('LunarLander-v2')
nn = DQN(8,4)
def runEp(env,printT=False):
    state = env.reset()
    state = np.reshape(state, [1, 8])
    totalreward = 0
    for _ in range(500):
        if printT:
            env.render()
        action = nn.chooseAction(state)
        next_state, reward, done, _ = env.step(action)
        totalreward +=reward
        next_state = np.reshape(next_state, [1, 8])
        nn.storePastResults(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
        nn.train(6)
    return totalreward

avgReward = np.zeros(100)
for i in range(20000): # Number of episodes s
    if i %100 == 0:
        avgReward[i%100] = runEp(env,True)
        avg = np.average(avgReward)
        print('Episode %f/10000 Reward: %f' %(i,avg))
    else:
        avgReward[i%100] =runEp(env,True)
        print(i,avgReward[i%100])














##
