"""Active loiter guidance path in sub_Guidance.slx and sub_UAV.slx."""

import numpy as np


GRAVITY = 9.81
ACS_BANK_LIMIT = np.deg2rad(30)
VF_BANK_LIMIT = np.deg2rad(40)
TARGET_SPEED = 25.0


def _alignment_commands(heading, airspeed, velocity, delta_north, delta_east, distance_sq):
    """Return ACS longitudinal acceleration and lateral acceleration."""
    count = len(heading)
    eta = (1.0 + distance_sq)**-0.25
    np.fill_diagonal(eta, 0)
    north_velocity = velocity[:, 0]
    east_velocity = velocity[:, 1]
    relative_dot = (
        (north_velocity[None, :] - north_velocity[:, None]) * delta_north
        + (east_velocity[None, :] - east_velocity[:, None]) * delta_east
    )
    distance = np.sqrt(distance_sq + 1e-6)
    spacing_radius = 1.2 * np.sqrt(count * 13.7)
    # Coincident pairs contribute no positional augmentation.
    radial_rate = np.divide(
        relative_dot, distance_sq, out=np.zeros_like(relative_dot), where=distance_sq > 0
    )
    nu = 0.27 * radial_rate + 0.045 / distance * (distance - 2 * spacing_radius)
    north = (nu * delta_north).sum(axis=1)
    east = (nu * delta_east).sum(axis=1)
    sin_heading, cos_heading = np.sin(heading), np.cos(heading)
    longitudinal = (eta * (airspeed[None, :] - airspeed[:, None])).sum(axis=1)
    lateral = (eta * airspeed[None, :] * np.sin(heading[None, :] - heading[:, None])).sum(axis=1)
    return (
        (longitudinal + north * cos_heading + east * sin_heading) / count,
        (lateral - north * sin_heading + east * cos_heading) / count,
    )


def _loiter_bank(positions, airspeed, velocity, center, orbit_radius):
    """Return the VF bank command for a circular orbit."""
    north, east = positions.mean(axis=0) - center
    radius = np.hypot(north, east)
    safe_radius = max(radius, 1e-3)
    orbit_sq = orbit_radius**2
    radius_sq = radius**2
    factor = -airspeed / (radius * (radius_sq + orbit_sq) + 1e-9)
    desired_north = factor * (north * (radius_sq - orbit_sq) + east * 2 * orbit_radius * radius)
    desired_east = factor * (east * (radius_sq - orbit_sq) - north * 2 * orbit_radius * radius)
    velocity_north, velocity_east = velocity.T
    heading_error = np.arctan2(desired_east, desired_north) - np.arctan2(velocity_east, velocity_north)
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    theta_dot = (north * velocity_east - east * velocity_north) / safe_radius**2
    radius_dot = (north * velocity_north + east * velocity_east) / safe_radius
    desired_turn = theta_dot + 2 * orbit_radius * radius_dot / (orbit_sq + safe_radius**2)
    return np.arctan2(airspeed * (desired_turn + heading_error), GRAVITY).clip(
        -VF_BANK_LIMIT, VF_BANK_LIMIT
    )


def guidance_commands(states, weights, center, orbit_radius, dt):
    """Return bank and airspeed commands. Weight columns are ACS, PF and VF."""
    positions = states[:, :2]
    heading = states[:, 2]
    airspeed = states[:, 3]
    velocity = states[:, 4:6]
    north, east = positions.T
    delta_north = north[None, :] - north[:, None]
    delta_east = east[None, :] - east[:, None]
    distance_sq = delta_north**2 + delta_east**2

    acc, turn = _alignment_commands(heading, airspeed, velocity, delta_north, delta_east, distance_sq)
    acs = np.arctan(turn / GRAVITY).clip(-ACS_BANK_LIMIT, ACS_BANK_LIMIT)

    # PF subsystem 4749 projects summed repulsion onto the lateral axis.
    pf_distance = np.sqrt(distance_sq) + 1e-9
    pf_scale = -8.0 / (1 + (pf_distance / 8.0)**10) / pf_distance
    pf_north = (pf_scale * delta_north).sum(axis=1)
    pf_east = (pf_scale * delta_east).sum(axis=1)
    pf = np.arctan2(-pf_north * np.sin(heading) + pf_east * np.cos(heading), GRAVITY)

    vf = _loiter_bank(positions, airspeed, velocity, center, orbit_radius)
    acs_weight, pf_weight, vf_weight = weights.T
    bank_command = acs_weight * acs + pf_weight * pf + vf_weight * vf
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    speed_command = TARGET_SPEED + dt * (acc + TARGET_SPEED - speed)
    return bank_command, speed_command
