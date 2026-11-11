"""Single-UAV discrete dynamics from sub_Dynamics.slx."""

from dataclasses import dataclass
from math import cos, pi, sin, sqrt, tan
from typing import NamedTuple


class UAVState(NamedTuple):
    """Simulink state vector in port order (m, rad, and m/s)."""

    north: float
    east: float
    heading: float
    airspeed: float
    air_velocity_north: float
    air_velocity_east: float
    bank: float


@dataclass(frozen=True)
class DynamicsParameters:
    """Defaults from simulink_param.m."""

    dt: float = 0.1
    airspeed_time_constant: float = 1.0
    bank_time_constant: float = 0.3
    min_airspeed: float = 18.0
    max_airspeed: float = 40.0


def step_dynamics(
    previous: UAVState,
    bank_command: float,
    airspeed_command: float,
    *,
    wind_north: float = 0.0,
    wind_east: float = 0.0,
    parameters: DynamicsParameters = DynamicsParameters(),
) -> tuple[UAVState, float]:
    """Return the next state and ground speed after one step.

    Bank command is in radians. Airspeed command and wind components are
    in m/s. Wind is disabled by default, as in simulink_param.m.
    """
    dt = parameters.dt
    bank = previous.bank + (1.0 / parameters.bank_time_constant) * (
        bank_command - previous.bank
    ) * dt
    airspeed = previous.airspeed + (1.0 / parameters.airspeed_time_constant) * (
        airspeed_command - previous.airspeed
    ) * dt
    airspeed = min(parameters.max_airspeed, max(parameters.min_airspeed, airspeed))

    heading = previous.heading + (9.81 / airspeed) * tan(bank) * dt
    if heading > pi:
        heading -= 2 * pi
    elif heading < -pi:
        heading += 2 * pi

    air_velocity_north = airspeed * cos(heading)
    air_velocity_east = airspeed * sin(heading)
    ground_velocity_north = air_velocity_north + wind_north
    ground_velocity_east = air_velocity_east + wind_east
    ground_speed = sqrt(ground_velocity_north**2 + ground_velocity_east**2)

    state = UAVState(
        north=previous.north + ground_velocity_north * dt,
        east=previous.east + ground_velocity_east * dt,
        heading=heading,
        airspeed=airspeed,
        air_velocity_north=air_velocity_north,
        air_velocity_east=air_velocity_east,
        bank=bank,
    )
    return state, ground_speed
