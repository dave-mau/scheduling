from typing import Tuple

import numpy as np
import pytest
from computation_sim.example_systems import SimpleChainBuilder
from computation_sim.nodes import Node
from computation_sim.system import System
from computation_sim.time import Clock, FixedDuration


@pytest.fixture
def setup() -> Tuple[System, Clock, Tuple[Node]]:
    clock = Clock(0)
    builder = SimpleChainBuilder(clock, sensor_epoch=0, sensor_period=100)
    builder.sensor_disturbance = FixedDuration(0)
    builder.compute_duration = FixedDuration(10)
    builder.build()

    return builder.system, clock, builder.nodes


def test_no_action_no_output(setup):
    system, clock, nodes = setup
    while clock.get_time() < 1000:
        clock += 1
        system.update()

    assert nodes["OUTPUT"].last_received is None


def test_nominal(setup):
    system, clock, nodes = setup
    # state vector:
    # [compute_busy, clock-time, num accepted inputs, num measurements total, output has result, ..output-times.., output-num-measurements, sink-count]

    # t: [0, 50) -- Wait and do nothing
    while clock.get_time() < 50:
        system.update()
        np.testing.assert_allclose(
            [0.0, float(clock.get_time()), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            np.array(system.state),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        clock += 1
        assert nodes["OUTPUT"].last_received is None

    # t = 50 -- ACT (start compute)
    system.act([1])

    # t: [50, 60) -- COMPUTE
    while clock.get_time() < 60:
        system.update()
        np.testing.assert_allclose(
            [1.0, float(clock.get_time() - 50), 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            np.array(system.state),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        clock += 1
        assert nodes["OUTPUT"].last_received is None

    # t = 60 -- OUTPUT RECEIVED
    system.update()
    np.testing.assert_allclose(
        [0.0, float(clock.get_time() - 50), 0.0, 0.0, 1.0, 60.0, 60.0, 60.0, 1.0, 0.0],
        np.array(system.state),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
