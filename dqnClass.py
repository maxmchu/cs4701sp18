import random
import keras as k
import numpy as np


class DQN:
    '''
    This aim of this class is to have a DQN framework that can be used for any
    open AI gym framework. Needs the function to train, and predict the next
    action given the state.

    '''
    def __init__(self,inputSize,outputSize,learningRate=.001,maxMem=1000):
        self.inputSize = inputSize # Number of possible states |state space|
        self.outputSize = outputSize # Number of possible actions
        self.learningRate = learningRate
        self.nnModel = self.buildNetwork()
        self.gamma = .95 # This is how much to care about future reward vs now
        self.exploreRate = 1.0
        self.exploreMin  = .01
        self.exploreDecay = .995
        self.maxMemory  = maxMem
        self.index = 0
        self.past = []

    def buildNetwork(self):
        # Neural Net for Deep-Q learning Model
        model = k.Sequential()
        model.add(k.layers.Dense(24, input_dim=self.inputSize, activation='relu'))
        model.add(k.layers.Dense(24, activation='relu'))
        model.add(k.layers.Dense(self.outputSize, activation='linear'))
        model.compile(loss='mse',optimizer=k.optimizers.Adam(lr=self.learningRate))
        return model

    def storePastResults(self, state, action, reward, next_state, done):
        if self.index > self.maxMemory:
            i = self.index % self.maxMemory
            self.past[i] = (state, action, reward, next_state, done)
        else:
            self.past.append((state, action, reward, next_state, done))
        self.index += 1

    def chooseAction(self,state):
        # Takes state and gives neural network action
        if np.random.rand() <= self.exploreRate:
            return random.randrange(self.outputSize)
        action = self.nnModel.predict(state)
        return np.argmax(action[0])

    def train(self,batchSize):
        # Takes all past results and trains
        batchSize = batchSize if batchSize <= len(self.past) else len(self.past)
        minibatch = random.sample(self.past, batchSize)
        for state, action, reward, next_state, done in minibatch:
            # So basically takes a state, changes the value to be the reward of
            # that action plus a discounted rate for the next step. So have
            # input state and reward for taking that action.
            state = np.reshape(state, [1, self.inputSize])
            target = reward
            if not done:
                target += self.gamma * np.amax(self.nnModel.predict(next_state)[0])
            target_f = self.nnModel.predict(state)
            target_f[0][action] = target
            self.nnModel.fit(state, target_f, epochs=1, verbose=0)
        if self.exploreRate > self.exploreMin:
            self.exploreRate*=self.exploreDecay



















#
