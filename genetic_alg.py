from wann_neuroev.evolution import Individual, HIDDEN, save_ind, Act
from copy import deepcopy
import gymnasium as gym
import numpy as np
import random
from tqdm import tqdm
import sys
import os
os.environ['MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT'] = '1'
import mlflow


# Воспроизводимость
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

""" ============ MLFLOW ============ """
EXPERIMENT_NAME = "NeuroEv_WANN_report"
RUN_SUFFIX = "_lunar_evolution"

""" ============ ГИПЕРПАРАМЕТРЫ ============ """
POPULATION_SIZE = 30
GENERATIONS_NUM = 300

START_CONS = 10 # Начальное кол-во связей у особи
N_SURVIVORS = 15 # Кол-во отобранных ранговым отбором
K_TOURNAMENT = 3 # Кол-во особей для турнира
MUTATE_RATE = 2 # Кол-во мутаций за раз
W_VALUES = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)

# add_con, add_node, change_act
MUTATE_PROBS = (0.25, 0.25, 0.5) # 0.25, 0.25, 0.5 - из статьи
INPUT_ACTIVATION = Act.linear
OUTPUT_ACTIVATION = Act.linear

TEST_EPISODES = 3
FINAL_TEST_EPISODES = 5

def init_population(pop_size, start_cons=4):
    population = []
    for _ in range(pop_size):
        ind = Individual(8, 4,
                         inp_act=INPUT_ACTIVATION, out_act=OUTPUT_ACTIVATION)
        for _ in range(start_cons):
            ind.mutate_add_con()
        population.append(ind)

    return population

def evaluate_individual(env, ind: Individual, weights: tuple, n_episodes=TEST_EPISODES):
    total_wins = 0
    total_reward = 0.0

    for w in weights:
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                ind_out = ind.forward(tuple(obs), w)
                action = ind_out.index(max(ind_out))
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                done = terminated or truncated

            total_reward += ep_reward
            if ep_reward >= 200:
                total_wins += 1

    avg_reward = total_reward / (len(weights) * n_episodes)
    avg_wins = total_wins / (len(weights) * n_episodes)

    return avg_reward, avg_wins

def rank_selection(env, population: dict[int, Individual]):
    avg_rewards = dict()
    winrates = dict()
    complexity = dict()

    for i_id, ind in population.items():
        ind_rew, ind_wins = evaluate_individual(env, ind, W_VALUES)
        avg_rewards[i_id] = ind_rew
        winrates[i_id] = ind_wins
        complexity[i_id] = len(ind.connections)

    # Победы/Награды/Сложность
    avg_rewards_sort = sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True)
    complexity = sorted(complexity.items(), key=lambda x: x[1])

    # Ранги
    reward_ranks = {i_id: rank for rank, (i_id, _) in enumerate(avg_rewards_sort)}
    total_ranks = {i_id: reward_ranks[i_id] for i_id in population}

    # Выжившие
    survivors = tuple(sorted(population.keys(), key=lambda i_id: total_ranks[i_id])[:N_SURVIVORS])
    # survivors = tuple(random.sample(list(population.keys()), N_SURVIVORS)) # BASELINE без селекии
    return survivors, total_ranks, avg_rewards, winrates

def mutate(ind: Individual, probs: tuple, mutate_num=1):
    for _ in range(mutate_num):
        mutate_vars = [ind.mutate_add_con, ind.mutate_add_node, ind.mutate_activation]
        mutation = random.choices(mutate_vars, weights=probs, k=1)[0]
        mutation() # Мутирование

""" ============ ЭВОЛЮЦИЯ ============ """
if __name__ == "__main__":
    # MLFLOW
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name=EXPERIMENT_NAME + RUN_SUFFIX):
        params = {
            "seed": SEED,
            "population_size": POPULATION_SIZE,
            "generations_number": GENERATIONS_NUM,
            "start_connections": START_CONS,
            "number_of_survivors": N_SURVIVORS,
            "k_tournament": K_TOURNAMENT,
            "mutations_num": MUTATE_RATE,
            "mutation_probs": MUTATE_PROBS,
            "w_values": W_VALUES,
            "test_episodes": TEST_EPISODES,
            "input_act": INPUT_ACTIVATION,
            "output_act": OUTPUT_ACTIVATION,
        }
        mlflow.log_params(params)

        env = gym.make("LunarLander-v3", render_mode=None)
        obs, _ = env.reset(seed=SEED)

        # Рождение жизни (~3.9 миллиарда лет назад)
        population: dict[int, Individual] = {i_id: ind for i_id, ind in
                                             enumerate(init_population(POPULATION_SIZE, START_CONS))}

        # Эволюция
        pbar = tqdm(range(1, GENERATIONS_NUM+1), desc="Evolving...", file=sys.stdout)
        for gen in pbar:
            # Ранговый отбор
            survivors, total_ranks, avg_rewards, winrates = rank_selection(env, population)

            # Логи
            best_id = min(total_ranks, key=lambda i_id: total_ranks[i_id])
            pop_avg_reward = sum(avg_rewards.values()) / len(avg_rewards)
            pop_winrate = sum(winrates.values()) / len(winrates)
            pop_connections = sum(len(ind.connections) for ind in population.values()) / len(population)
            pop_hidden = sum(len(ind.get_node_ids(HIDDEN)) for ind in population.values()) / len(population)

            # Лучший из лучших
            best_reward = avg_rewards[best_id]
            best_winrate = winrates[best_id]

            pbar.set_postfix(pop_winrate=f"{pop_winrate:.2f}",
                             pop_reward=f"{pop_avg_reward:.2f}",
                             best_winrate=f"{best_winrate:.2f}",
                             best_reward=f"{best_reward:.2f}",)

            mlflow.log_metrics({
                "pop_winrate": pop_winrate,
                "pop_rewards": pop_avg_reward,
                "pop_cons": pop_connections,
                "pop_hidden": pop_hidden,
                "best_reward": best_reward,
                "best_winrate": best_winrate,
            }, step=gen)

            """ ЧЕКПОИНТЫ (ОЦЕНКА РАЗВИТИЯ) """
            if gen % 50 == 0 or gen == 1:
                best_id = min(total_ranks, key=lambda i_id: total_ranks[i_id])
                best_ind = population[best_id]
                save_ind(best_ind, f'ind_checkpoints/best_ind_gen_{gen}.json')

            # Создание новой популяции
            new_population = {new_id: population[i_id] for new_id, i_id in enumerate(survivors)}
            while len(new_population) < POPULATION_SIZE:
                # Турнирный отбор
                rivals = random.sample(survivors, K_TOURNAMENT)
                winner_id = min(rivals, key=lambda i_id: total_ranks[i_id])

                # Мутация потомства
                new_ind = deepcopy(population[winner_id])
                mutate(new_ind, MUTATE_PROBS, MUTATE_RATE)

                new_population[len(new_population)] = new_ind

            population = new_population

        # Финальные особи
        survivors, total_ranks, avg_rewards, winrates = rank_selection(env, population)

        best_id = min(total_ranks, key=lambda i_id: total_ranks[i_id])
        best_ind = population[best_id]

        # Лучшая особь
        best_w = max(W_VALUES, key=lambda w: evaluate_individual(env, best_ind, (w,),
                                                                 n_episodes=FINAL_TEST_EPISODES)[1])
        best_reward, best_winrate = evaluate_individual(env, best_ind, (best_w,),
                                                        n_episodes=FINAL_TEST_EPISODES)
        env.close()

        # Отчёт по лучшей особи
        report = (
            f"Параметры лучшей особи:\n"
            f"Процент побед (средний, все веса): {winrates[best_id]}\n"
            f"Ср.награда (все веса): {avg_rewards[best_id]}\n"
            f"Лучший вес: {best_w}\n"
            f"Процент побед (лучший вес): {best_winrate}\n"
            f"Ср.награда (лучший вес): {best_reward}\n"
            f"Кол-во связей: {len(best_ind.connections)}\n"
            f"Кол-во скрытых нейронов: {len(best_ind.get_node_ids(HIDDEN))}"
        )

        with open("test_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact("test_report.txt")
        print(report)

        save_ind(best_ind, 'best_ind.json')
        mlflow.log_artifact(f"best_ind.json")