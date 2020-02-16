import gym
import gym_sokoban
import time

env = gym.make('Sokoban-v0')
env.reset()

for _ in range(1000):
    env.render(mode='human')
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        env.render(mode='human')
        break
    time.sleep(0.1)
