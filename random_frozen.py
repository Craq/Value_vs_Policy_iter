import numpy as np
import time
import gym

def execute(env,policy,episodeLength=100,render=False):
	total_reward = 0
	start = env.reset()
	for t in range(episodeLength):
		if render:
			env.render()
		action = policy[start]
		start,reward,done,_ = env.step(action)
		total_reward += reward
		if done:
			break
	return total_reward

def evaluatePolicy(env,policy, n_episodes=100):
	total_reward = 0.0
	for _ in range(n_episodes):
		total_reward += execute(env,policy)
	return total_reward / n_episodes

def get_random_policy():
	return np.random.choice(4,size=16)

if __name__=='__main__':
	env=gym.make('FrozenLake-v0')
	n_policies = 1000
	startTime = time.time()
	policy_set = [get_random_policy() for _ in range(n_policies)]
	policy_score = [evaluatePolicy(env,p)for p in policy_set]
	endTime=time.time()
	print('Best score = %r. Time taken = %4.4f sec'%(np.max(policy_score),endTime-startTime))