"""Run the swarm with fixed guidance weights and plot its trajectory."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from swarm_env import CENTER, ORBIT_RADIUS, UAV_COUNT, SwarmEnv


def save_animation(positions, times, output):
    centroid = positions.mean(axis=1)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.add_patch(plt.Circle(CENTER[::-1], ORBIT_RADIUS, fill=False,
                            linestyle="--", color="tab:red", label="Reference orbit"))
    lower = np.minimum(positions.min(axis=(0, 1)), CENTER - ORBIT_RADIUS) - 30
    upper = np.maximum(positions.max(axis=(0, 1)), CENTER + ORBIT_RADIUS) + 30
    ax.set(xlim=(lower[1], upper[1]), ylim=(lower[0], upper[0]),
           xlabel="East (m)", ylabel="North (m)", aspect="equal")
    tracks = [ax.plot([], [], marker=".", markevery=[-1], label=f"UAV {i + 1}")[0]
              for i in range(UAV_COUNT)]
    center_track, = ax.plot([], [], color="black", marker=".", markevery=[-1], label="Centroid")
    ax.legend(fontsize=8)
    frames = np.unique(np.linspace(0, len(positions) - 1, min(len(positions), 201), dtype=int))

    def draw(frame):
        for line, track in zip(tracks, positions[:frame + 1].transpose(1, 0, 2)):
            line.set_data(track[:, 1], track[:, 0])
        center_track.set_data(centroid[:frame + 1, 1], centroid[:frame + 1, 0])
        ax.set_title(f"Simulation time {times[frame]:.1f} s")

    animation = FuncAnimation(fig, draw, frames=frames)
    animation.save(output, writer=PillowWriter(fps=20), dpi=85)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--weights", type=float, nargs=3, default=(1, 1, 1),
                        metavar=("ACS", "PF", "VF"))
    parser.add_argument("--output", type=Path, default=Path("runs/trajectory.png"))
    parser.add_argument("--animate", action="store_true", help="also save an animated GIF")
    args = parser.parse_args()
    weights = np.asarray(args.weights)
    if args.steps < 1 or not np.all((0 <= weights) & (weights <= 1)):
        parser.error("steps must be positive and weights must be between 0 and 1")

    env = SwarmEnv(args.steps, enforce_action_bounds=False)
    env.reset()
    positions = [env.states[:, :2].copy()]
    actions = np.broadcast_to(weights, (UAV_COUNT, 3))
    # The fixed controller accepts zero weights to isolate each guidance term.
    # Preserve the original initial zero command and one-step action delay.
    for _ in range(args.steps):
        env.step(actions)
        positions.append(env.states[:, :2].copy())
    positions = np.array(positions)
    centroid = positions.mean(axis=1)
    error = np.linalg.norm(centroid - CENTER, axis=1) - ORBIT_RADIUS
    times = np.arange(len(positions)) * env.parameters.dt

    fig, (path_ax, error_ax) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for i, track in enumerate(positions.transpose(1, 0, 2), start=1):
        path_ax.plot(track[:, 1], track[:, 0], alpha=0.7, label=f"UAV {i}")
    path_ax.plot(centroid[:, 1], centroid[:, 0], color="black", linewidth=1.5, label="Centroid")
    path_ax.scatter(*centroid[0, ::-1], color="black", marker="o", label="Start")
    path_ax.scatter(*centroid[-1, ::-1], color="black", marker="x", label="End")
    path_ax.add_patch(plt.Circle(CENTER[::-1], ORBIT_RADIUS, fill=False,
                                 linestyle="--", color="tab:red", label="Reference orbit"))
    path_ax.set(xlabel="East (m)", ylabel="North (m)", title="Swarm flight paths", aspect="equal")
    path_ax.legend(fontsize=8)
    error_ax.plot(times, error, color="black")
    error_ax.axhline(0, linestyle="--", color="tab:red")
    error_ax.set(xlabel="Simulation time (s)", ylabel="Signed radial error (m)",
                 title="Centroid distance from reference orbit")
    error_ax.grid(alpha=0.25)
    fig.suptitle(f"Fixed guidance weights   ACS={weights[0]:g}, PF={weights[1]:g}, VF={weights[2]:g}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    if args.animate:
        save_animation(positions, times, args.output.with_suffix(".gif"))
    print(f"Centroid radial error: {error[0]:.2f} m initially, {error[-1]:.2f} m at {times[-1]:g} s")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
