import numpy as np

from ..system_fixtures import simple_chain


def test_no_action_no_output(simple_chain):
    system, clock, nodes = simple_chain
    while clock.get_time() < 1000:
        clock += 1
        system.update()

    assert nodes[2].last_received is None


def test_nominal(simple_chain):
    system, clock, nodes = simple_chain
    # state vector:
    # [compute_busy, clock-time, output has result, ..output-times.., sink-count]

    # t: [0, 50) -- Wait and do nothing
    while clock.get_time() < 50:
        system.update()
        np.testing.assert_allclose(
            [0.0, float(clock.get_time()), 0.0, 0.0, 0.0, 0.0, 0.0],
            np.array(system.state),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        clock += 1
        assert nodes[2].last_received is None

    # t = 50 -- ACT (start compute)
    system.act([1])

    # t: [50, 60) -- COMPUTE
    while clock.get_time() < 60:
        system.update()
        np.testing.assert_allclose(
            [1.0, float(clock.get_time() - 50), 0.0, 0.0, 0.0, 0.0, 0.0],
            np.array(system.state),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        clock += 1
        assert nodes[2].last_received is None

    # t = 60 -- OUTPUT RECEIVED
    system.update()
    np.testing.assert_allclose(
        [0.0, float(clock.get_time() - 50), 1.0, 60.0, 60.0, 60.0, 0.0],
        np.array(system.state),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
