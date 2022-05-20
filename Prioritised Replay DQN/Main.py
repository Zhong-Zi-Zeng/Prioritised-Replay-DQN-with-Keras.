import gym
import numpy as np
from DQN import Agent
import matplotlib.pyplot as plt

EPISODE = 20


def main(agent):
    total_steps = 0
    steps = []
    episodes = []

    for i in range(EPISODE):
        done = False
        state = env.reset()
        print('now episode is: {}'.format(i))
        while not done:
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            if done:
                reward = 10

            agent.remember(state,action,reward,next_state,done)

            if total_steps > 8192:
                agent.learn()

            state = next_state
            total_steps += 1

        steps.append(total_steps)
        episodes.append(i)

    return np.array(episodes), np.array(steps)

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    pri_agent = Agent(alpha=0.0005,
                  gamma=0.9,
                  n_actions=3,
                  epsilon=0.7,
                  batch_size=32,
                  epsilon_end=0.1,
                  epsilon_dec=0.9,
                  mem_size=8192,
                  iteration=20,
                  input_shape=2,
                  use_pri=True)

    ori_agent = Agent(alpha=0.0005,
                  gamma=0.9,
                  n_actions=3,
                  epsilon=0.7,
                  batch_size=32,
                  epsilon_end=0.1,
                  epsilon_dec=0.95,
                  mem_size=8192,
                  iteration=20,
                  input_shape=2,
                  use_pri=False)

    use_pri = main(pri_agent)
    not_use_pri = main(ori_agent)

    plt.plot(use_pri[0],use_pri[1] - use_pri[1][0], label='use_pri')
    plt.plot(not_use_pri[0], not_use_pri[1] - not_use_pri[1][0], label='not_use_pri')
    plt.legend()
    plt.grid()
    plt.show()

