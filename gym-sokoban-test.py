import gym
import gym_sokoban

env = gym.make('Sokoban-v0')

env.render(mode='human')

action = env.action_space.sample()
observation, reward, done, info = env.step(action)
