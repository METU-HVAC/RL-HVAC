from algorithms.ddqn.ddqn import Agent
import gymnasium as gym
import numpy as np

env = gym.make('LunarLanderContinuous-v2')

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=4)

#agent.load_models()
np.random.seed(0)

score_history = []
for i in range(1000):
    obs,_= env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        print(act)
        new_state, reward, truncated,terminated, info = env.step(act)
        done = terminated or truncated
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        #env.render()
    score_history.append(score)

    #if i % 25 == 0:
    #    agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

