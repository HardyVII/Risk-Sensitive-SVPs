import numpy as np
import matplotlib.pyplot as plt
from lifegate_for_SVPs import LifeGate
from SVPs_algos import value_iter, V2Q, zeta_optimal_SVP, value_iter_near_greedy, value_iter_near_greedy_prob


def MDP_lifegate(env, types='regular'):
    P = {}
    width, height = env.scr_w, env.scr_h
    for y in range(height):
        for x in range(width):

            s = y * width + x
            P[s] = {}

            # Barriers got no list
            if  [x, y] in env.barriers:
                continue

            for a in env.legal_actions:
                P[s][a] = []
                new_x, new_y = x, y
                reward = 0.0
                done = False

                # Dead_ends state can always move to right without getting out of bound.
                if [x, y] in env.dead_ends:
                    if [x + 1, y] in env.deaths:
                        done = True
                        if types == 'death' or types == 'regular':
                            reward = -1.0
                    P[s][a].append((0.7, s + 1, reward, done))
                    P[s][a].append((0.3, s, 0, False))
                    continue

                # If the current state is termination, quick set up and continue.
                if [new_x, new_y] in env.deaths:
                    if types == 'death' or types == 'regular':
                        reward = -1.0
                    P[s][a].append((1.0, s, reward, True))
                    continue
                if [new_x, new_y] in env.recoveries:
                    if types == 'recovery' or types == 'regular':
                        reward = 1.0
                    P[s][a].append((1.0, s, reward, True))
                    continue

                # Move the player.
                if a == 1:
                    new_y = y - 1
                elif a == 2:
                    new_y = y + 1
                elif a == 3:
                    new_x = x - 1
                elif a == 4:
                    new_x = x + 1

                # If the player get out of bound, move it back to the previous state.
                if new_x < 0 or new_y < 0 or new_x >= width or new_y >= height or [new_x, new_y] in env.barriers:
                    new_x, new_y = x, y

                # Set up for regular states, regular states also have a natural risk to be dragged to the right.
                reward_drag = 0.0
                done_drag = False
                s_next = new_y * width + new_x
                s_drag = s + 1

                if [new_x, new_y] in env.deaths:
                    done = True
                    if types == 'death' or types == 'regular':
                        reward = -1.0
                elif [new_x, new_y] in env.recoveries:
                    done = True
                    if types == 'recovery' or types == 'regular':
                        reward = 1.0

                if [x + 1, y] in env.deaths:
                    done_drag = True
                    if types == 'death' or types == 'regular':
                        reward_drag = -1.0

                P[s][a].append((1 - env.death_drag, s_next, reward, done))
                P[s][a].append((env.death_drag, s_drag, reward_drag, done_drag))
    return  P


def visualize_v_grid(grid, title="V-Grid Heatmap", cmap="RdBu", vmin=None, vmax=None):
    plt.imshow(grid, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax)
    plt.colorbar(label="Value")
    plt.title(title)
    plt.axis("off")
    plt.show()


def main():
    # random seed
    random_state = np.random.RandomState(1234)

    # Initial setup for set-valued-policies environment.
    env = LifeGate(state_mode='tabular', rng=random_state, death_drag=0.4, fixed_life=True)
    env.P = MDP_lifegate(env)
    env.nS = env.scr_w * env.scr_h  # number of states (10*10)
    env.nA = env.nb_actions  # number of actions (should be 5)
    # 1) Compute the optimal value function and policies using value iteration.
    V_star, π_star = value_iter(env=env, gamma=1)
    # 2) Compute the worst value function and near-optimal policies using near greedy
    V_SVP, π_SVP = value_iter_near_greedy_prob(env=env, gamma=1, rho=0.1,
                                               V_star=V_star, zeta=0.1, theta=1e-10, max_iter=1000)


    # Initial setup for lifegate environment.
    env_lifegate = LifeGate(state_mode='tabular', rng=random_state, death_drag=0.4, fixed_life=True)
    env_lifegate.P = MDP_lifegate(env, types='recovery')
    env_lifegate.nS = env_lifegate.scr_w * env_lifegate.scr_h
    env_lifegate.nA = env_lifegate.nb_actions
    # 3) Compute the V_r and r-policies.
    V_r, π_r = value_iter(env=env_lifegate, gamma=1)


    # Initial setup for death environment.
    env_death = LifeGate(state_mode='tabular', rng=random_state, death_drag=0.4, fixed_life=True)
    env_death.P = MDP_lifegate(env, types='death')
    env_death.nS = env_death.scr_w * env_death.scr_h
    env_death.nA = env_death.nb_actions
    # 4) Compute the V_d and d-policies.
    V_d, π_d = value_iter(env=env_death, gamma=1)


    # Visualize the three V-functions.
    V_SVP_Grid = np.zeros((env.scr_w, env.scr_h))
    V_r_Grid = np.zeros((env.scr_w, env.scr_h))
    V_d_Grid = np.zeros((env.scr_w, env.scr_h))
    for y in range(env.scr_h):
        for x in range(env.scr_w):
            V_SVP_Grid[y, x] = V_SVP[y * env.scr_w + x]
            V_r_Grid[y, x] = V_r[y * env.scr_w + x]
            V_d_Grid[y, x] = V_d[y * env.scr_w + x]

    visualize_v_grid(V_SVP_Grid, title="V_SVP Grid", vmin=V_SVP.min(), vmax=V_SVP.max())
    visualize_v_grid(V_r_Grid, title="V_R Grid", vmin=V_r.min(), vmax=V_r.max())
    visualize_v_grid(V_d_Grid, title="V_D Grid", vmin=V_d.min(), vmax=V_d.max())


if __name__ == '__main__':
    main()
