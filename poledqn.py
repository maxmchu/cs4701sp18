import gym
import numpy as np
from dqnClass import DQN

# Setup Enviornment
env = gym.make('CartPole-v0')
nn = DQN(4,2)
def runEp(env,printT=False):
    state = env.reset()
    state = np.reshape(state, [1, 4])
    totalreward = 0
    for t in range(500):
        if printT:
            env.render()
        action = nn.chooseAction(state)
        next_state, reward, done, _ = env.step(action)
        totalreward +=reward
        next_state = np.reshape(next_state, [1, 4])
        nn.storePastResults(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
        nn.train(32)
    return t

avgReward = np.zeros(200)
for i in range(10000): # Number of episodes s
    score =runEp(env,True)
    print ('Episode %f, Score: %f' %(i,score))















##
