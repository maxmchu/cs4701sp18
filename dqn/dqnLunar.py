import gym
import numpy as np
from dqnClass import DQN


# Setup Enviornment
class DQNLunar:
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        self.nn = DQN(8,4)
        self.avgReward = np.zeros(100)

    def runEp(self, env, printT=False):
        state = env.reset()
        state = np.reshape(state, [1, 8])
        totalreward = 0
        for _ in range(500):
            if printT:
                env.render()
            action = self.nn.chooseAction(state)
            next_state, reward, done, _ = env.step(action)
            totalreward +=reward
            next_state = np.reshape(next_state, [1, 8])
            self.nn.storePastResults(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            self.nn.train(6)
        return totalreward


    def run(self):
        for i in range(20000): # Number of episodes s
            if i %100 == 0:
                self.avgReward[i%100] = self.runEp(self.env,True)
                avg = np.average(self.avgReward)
                print('Episode %f/10000 Reward: %f' %(i,avg))
            else:
                self.avgReward[i%100] = self.runEp(self.env,True)
                print(i,self.avgReward[i%100])














##
