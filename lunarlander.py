import gym
import numpy as np

# Setup Enviornment
env = gym.make('LunarLander-v2')

def runEp(env,pars1,pars2,printT):
    observation = env.reset()
    totalreward = 0
    for _ in xrange(400):
        if printT:
            env.render()
        policy = np.matmul(np.matmul(pars1,pars2),observation)
        action = np.argmax(policy)
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


parameters1 = np.random.rand(4,100)*2-1
parameters2 =np.random.rand(100,8)*2-1
noiseLevel = 1000.
reward = -1000
currentReward = reward
noiseLevel = 10000.
for i in range(1000):
    currentPars1 = parameters1 +(np.random.rand(4,100)*2-1)*noiseLevel
    currentPars2 = parameters2 +(np.random.rand(100,8)*2-1)*noiseLevel
    # average over 100 runs
    vals = np.zeros(100)
    for t in range(10):
        vals[t]= runEp(env,currentPars1,currentPars2,True)
    currentReward = np.average(vals)
    if currentReward > reward:
        print currentReward
        reward = currentReward
        parameters1 = currentPars1
        parameters2 = currentPars2
    if currentReward > 50:
        break


for i in range(10000):
    print runEp(env,parameters1,parameters2,True)













##
