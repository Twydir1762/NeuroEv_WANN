from wann_neuroev.evolution import load_ind
from wann_neuroev.visualizer import Visualizer
import gymnasium as gym

W = -2.0
SEED = 42
NUM_EPISODES = 100

best_ind = load_ind("best_ind.json")

vis = Visualizer((800, 600), 30, n_radius=18)
vis.draw(best_ind, column_size=10)

def test_ind(env):
    obs, _ = env.reset(seed=SEED)

    wins = 0
    for _ in range(NUM_EPISODES):
        obs, _ = env.reset()
        ep_reward = 0

        while True:
            ind_out = best_ind.forward(tuple(obs), W)
            action = ind_out.index(max(ind_out))

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            if terminated or truncated:
                if ep_reward >= 200:
                    wins += 1
                break

    env.close()
    return wins

env1 = gym.make("LunarLander-v3", render_mode=None, enable_wind=False)
no_wind_wins = test_ind(env1)

env2 = gym.make("LunarLander-v3", render_mode=None, enable_wind=True, wind_power=5.0)
wind_wins = test_ind(env2)

print(f"no_wind_winrate: {(no_wind_wins/NUM_EPISODES)*100}%")
print(f"wind_winrate: {(wind_wins/NUM_EPISODES)*100}%")