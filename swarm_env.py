"""Five-UAV environment for the fixed-target loiter training scenario."""

import numpy as np

from dynamics import DynamicsParameters, UAVState, step_dynamics
from guidance import guidance_commands


UAV_COUNT = 5
OBSERVATION_SIZE = 6
ACTION_LOWER = np.array([0.1, 0.0, 0.2])
CENTER = np.array([1000.0, 1000.0])
ORBIT_RADIUS = 250.0
INITIAL_HEADING_DEGREES = (66.50667771, 118.35539411, -45.35388853, 79.48694558, 57.75062730)
FORM_SPACING = 2 * 1.2 * np.sqrt(UAV_COUNT * 13.7)


def observations_and_rewards(states, orbit_scale):
    """Translate the active Observations.m and Rewards.m functions."""
    positions = states[:, :2]
    # RL_swarm_env StateReshape passes [N, E, Vt, psi] to the .m
    # functions. Their third row is consumed as heading.
    heading = states[:, 3]
    centroid = positions.mean(axis=0)
    orbit_error = np.linalg.norm(centroid - CENTER) - ORBIT_RADIUS
    orbit_error /= max(orbit_scale, 1.5)

    headings = np.column_stack((np.cos(heading), np.sin(heading)))
    mean_heading = headings.mean(axis=0)
    magnitude = np.linalg.norm(mean_heading)
    mean_heading = mean_heading / magnitude if magnitude > 1e-6 else np.array([1.0, 0.0])
    difference = np.arctan2(mean_heading[1], mean_heading[0]) - heading
    heading_error = np.arctan2(np.sin(difference), np.cos(difference))

    north, east = positions.T
    distances = np.hypot(north[:, None] - north, east[:, None] - east)
    np.fill_diagonal(distances, np.inf)
    nearest, second_nearest = np.partition(distances, kth=1, axis=1)[:, :2].T
    centroid_distance = np.linalg.norm(positions - centroid, axis=1)
    observations = np.column_stack((
        np.full(UAV_COUNT, np.tanh(orbit_error)),
        heading_error / np.pi,
        np.tanh((nearest - FORM_SPACING) / FORM_SPACING),
        np.tanh(centroid_distance / FORM_SPACING),
        np.exp(-nearest / 8),
        np.exp(-second_nearest / 8),
    ))
    rewards = (
        -abs(orbit_error)
        -np.var(centroid_distance, ddof=1) / FORM_SPACING**2
        -2 * (np.maximum(8 - nearest, 0) / 8)**2
        -(1 - headings @ mean_heading)
    )
    return observations.astype(np.float32), rewards.astype(np.float32)


class SwarmEnv:
    def __init__(self, max_steps=2000, *, enforce_action_bounds=True):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        self.action_lower = ACTION_LOWER if enforce_action_bounds else 0.0
        self.parameters = DynamicsParameters()
        self.steps = max_steps

    def reset(self, initial_states=None):
        """Return five observations. Optionally supply a MATLAB (5, 7) state array."""
        if initial_states is None:
            heading = np.deg2rad(INITIAL_HEADING_DEGREES)
            initial_states = np.column_stack((
                np.arange(-280, -180, 20),
                np.full(UAV_COUNT, 100),
                heading,
                np.full(UAV_COUNT, 20),
                20 * np.cos(heading),
                20 * np.sin(heading),
                np.zeros(UAV_COUNT),
            ))
        self.states = np.array(initial_states, dtype=float, copy=True)
        if self.states.shape != (UAV_COUNT, 7) or not np.isfinite(self.states).all():
            raise ValueError("initial_states must be a finite (5, 7) array")
        if np.any(self.states[:, 3] <= 0):
            raise ValueError("Initial airspeeds must be positive")
        self.orbit_scale = abs(np.linalg.norm(self.states[:, :2].mean(axis=0) - CENTER) - ORBIT_RADIUS)
        self.steps = 0
        self.pending_actions = np.zeros((UAV_COUNT, 3))
        observations, _ = observations_and_rewards(self.states, self.orbit_scale)
        return observations

    def _advance(self, actions):
        bank, speed = guidance_commands(
            self.states, actions, CENTER, ORBIT_RADIUS, self.parameters.dt
        )
        self.states = np.array([
            step_dynamics(UAVState(*previous), bank_cmd, speed_cmd, parameters=self.parameters)[0]
            for previous, bank_cmd, speed_cmd in zip(self.states, bank, speed)
        ])

    def step(self, actions):
        """Queue (5, 3) actions and advance with the previously queued actions."""
        if self.steps >= self.max_steps:
            raise RuntimeError("Reset the environment before starting another episode")
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (UAV_COUNT, 3) or not np.isfinite(actions).all():
            raise ValueError("actions must be a finite (5, 3) array")
        self._advance(self.pending_actions)
        self.pending_actions = actions.clip(self.action_lower, 1)
        self.steps += 1
        observations, rewards = observations_and_rewards(self.states, self.orbit_scale)
        return observations, rewards, self.steps >= self.max_steps
