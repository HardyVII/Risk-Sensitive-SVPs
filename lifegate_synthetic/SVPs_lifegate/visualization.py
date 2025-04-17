import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lifegate_for_SVPs import LifeGate
from SVPs_algos import value_iter, value_iter_near_greedy_prob, V2Q, value_iter_near_greedy


def visualize_v_grid(grid, title="V-Grid Heatmap", cmap="RdBu", vmin=None, vmax=None):
    plt.imshow(grid, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax)
    plt.colorbar(label="Value")
    plt.title(title)
    plt.axis("off")
    plt.show()


def visualize_svp(π, barrier_states, lifegate_states, dead_end, deads_states, title="Set-Valued Policy"):
    grid_size = 10
    fig, ax = plt.subplots(figsize=(6, 6))
    arrow_offsets = {
        0: (0, 0),
        1: (0, -0.4),
        2: (0, 0.4),
        3: (-0.4, 0),
        4: (0.4, 0),
    }
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(-0.5, grid_size, 1))
    ax.set_yticks(np.arange(-0.5, grid_size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='major', color='black', linewidth=1)
    arrow_style = dict(arrowstyle="->", color="green", lw=2)
    for y in range(grid_size):
        for x in range(grid_size):
            s = y * grid_size + x
            if s in barrier_states:
                continue
            for a in range(5):
                if π[s, a] == 1:
                    dx, dy = arrow_offsets[a]
                    if a == 0:
                        ax.plot(x, y, 'o', markersize=4, color="green")
                    else:
                        x_end = x + dx
                        y_end = y + dy
                        ax.annotate("", xy=(x_end, y_end), xytext=(x, y), arrowprops=arrow_style)
    for s in barrier_states:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="gray", edgecolor="none")
        ax.add_patch(rect)
    for s in lifegate_states:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="blue", edgecolor="none")
        ax.add_patch(rect)
    for s in dead_end:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="yellow", edgecolor="none")
        ax.add_patch(rect)
    for s in deads_states:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="red", edgecolor="none")
        ax.add_patch(rect)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def visualize_ded(Q_d, barrier_states, lifegate_states, dead_end, deads_states, title="Dead-End Policy"):
    grid_size = 10
    fig, ax = plt.subplots(figsize=(6, 6))
    arrow_offsets = {
        0: (0, 0),
        1: (0, -0.4),
        2: (0, 0.4),
        3: (-0.4, 0),
        4: (0.4, 0),
    }
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(-0.5, grid_size, 1))
    ax.set_yticks(np.arange(-0.5, grid_size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='major', color='black', linewidth=1)
    arrow_style = dict(arrowstyle="->", color="red", lw=2)
    for y in range(grid_size):
        for x in range(grid_size):
            s = y * grid_size + x
            for a in range(5):
                if Q_d[s, a] <= -0.7:
                    dx, dy = arrow_offsets[a]
                    if a == 0:
                        ax.plot(x, y, 'ko', markersize=4)
                    else:
                        x_end = x + dx
                        y_end = y + dy
                        ax.annotate("", xy=(x_end, y_end), xytext=(x, y), arrowprops=arrow_style)
    for s in barrier_states:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="gray", edgecolor="none")
        ax.add_patch(rect)
    for s in lifegate_states:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="blue", edgecolor="none")
        ax.add_patch(rect)
    for s in dead_end:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="yellow", edgecolor="none")
        ax.add_patch(rect)
    for s in deads_states:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="red", edgecolor="none")
        ax.add_patch(rect)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def load_results(filename="trained_results.pkl"):
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results


def main():
    # Load the training results from file
    results = load_results("results/trained_results.pkl")

    # Extract components
    V_star = results["V_star"]
    π_star = results["π_star"]
    V_SVP = results["V_SVP"]
    π_SVP = results["π_SVP"]
    V_r = results["V_r"]
    π_r = results["π_r"]
    V_d = results["V_d"]
    π_d = results["π_d"]
    Q_d = results["Q_d"]

    # Create grids for V functions
    grid_shape = (10, 10)
    V_SVP_Grid = np.zeros(grid_shape)
    V_r_Grid = np.zeros(grid_shape)
    V_d_Grid = np.zeros(grid_shape)
    for y in range(10):
        for x in range(10):
            s = y * 10 + x
            V_SVP_Grid[y, x] = V_SVP[s]
            V_r_Grid[y, x] = V_r[s]
            V_d_Grid[y, x] = V_d[s]

    # Define special states
    barrier_states = [0, 1, 2, 3, 4, 51, 52, 53, 54]
    lifegate_states = [5, 6, 7]
    dead_ends = [45, 46, 47, 48, 55, 56, 57, 58, 65, 66, 67, 68, 75, 76, 77, 78, 85, 86, 87, 88, 95, 96, 97, 98]
    deads_states = [8, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

    # Visualize V function grids
    visualize_v_grid(V_SVP_Grid, title="V_SVP Grid", vmin=V_SVP.min(), vmax=V_SVP.max())
    visualize_v_grid(V_r_Grid, title="V_R Grid", vmin=V_r.min(), vmax=V_r.max())
    visualize_v_grid(V_d_Grid, title="V_D Grid", vmin=V_d.min(), vmax=V_d.max())

    # Visualize policies
    visualize_svp(π_SVP, barrier_states, lifegate_states, dead_ends, deads_states, title="Set-Valued Policy (SVP)")
    visualize_ded(Q_d, barrier_states, lifegate_states, dead_ends, deads_states,
                  title="Dead-End Policy")


if __name__ == "__main__":
    main()
