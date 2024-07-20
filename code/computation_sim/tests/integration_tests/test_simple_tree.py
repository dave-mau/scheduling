import numpy as np

from ..system_fixtures import simple_tree


def test_no_action(simple_tree):
    system, clock, _ = simple_tree

    # Run clock for first 99 time units
    while clock.get_time() < 100:
        # On first update, sensor immediately writes output to buffers.
        system.update()
        age = clock.get_time() / 100.0
        expected_state = np.array(
            [
                1.0,  #  0: Buffer 0: Occupancy
                age,  #  1: Buffer 0: Oldest Age
                age,  #  2: Buffer 0: Youngest Age
                age,  #  3: Buffer 0: Average Age
                1.0,  #  4: Buffer 1: Occupancy
                age,  #  5: Buffer 1: Oldest Age
                age,  #  6: Buffer 1: Youngest Age
                age,  #  7: Buffer 1: Average Age
                0.0,  #  8: Compute Node: Busy
                age,  #  9: Compute Node: Age last started
                0.0,  # 10: Lost Messages: Count
                0.0,  # 11: Output: Occupancy
                0.0,  # 12: Output: Oldest Age
                0.0,  # 13: Output: Youngest Age
                0.0,  # 14: Output: Average Age
            ]
        )
        state = np.array(system.state)
        np.testing.assert_allclose(expected_state, state)
        clock += 10

    # Sensors trigger at t=100.
    while clock.get_time() < 200:
        # Upon update, buffers will receive new messages.
        # The overflow (old) messages should be written to the lost-node.
        system.update()
        age = (clock.get_time() - 100) / 100.0
        expected_state = np.array(
            [
                1.0,  #  0: Buffer 0: Occupancy
                age,  #  1: Buffer 0: Oldest Age
                age,  #  2: Buffer 0: Youngest Age
                age,  #  3: Buffer 0: Average Age
                1.0,  #  4: Buffer 1: Occupancy
                age,  #  5: Buffer 1: Oldest Age
                age,  #  6: Buffer 1: Youngest Age
                age,  #  7: Buffer 1: Average Age
                0.0,  #  8: Compute Node: Busy
                age + 1.0,  #  9: Compute Node: Age last started
                2.0,  # 10: Lost Messages: Count
                0.0,  # 11: Output: Occupancy
                0.0,  # 12: Output: Oldest Age
                0.0,  # 13: Output: Youngest Age
                0.0,  # 14: Output: Average Age
            ]
        )
        state = np.array(system.state)
        np.testing.assert_allclose(expected_state, state)
        clock += 10


def test_nominal(simple_tree):
    system, clock, _ = simple_tree

    # t in [0, 50): Nothing
    while clock.get_time() < 50:
        system.update()
        age = clock.get_time() / 100.0
        expected_state = np.array(
            [
                1.0,  #  0: Buffer 0: Occupancy
                age,  #  1: Buffer 0: Oldest Age
                age,  #  2: Buffer 0: Youngest Age
                age,  #  3: Buffer 0: Average Age
                1.0,  #  4: Buffer 1: Occupancy
                age,  #  5: Buffer 1: Oldest Age
                age,  #  6: Buffer 1: Youngest Age
                age,  #  7: Buffer 1: Average Age
                0.0,  #  8: Compute Node: Busy
                age,  #  9: Compute Node: Age last started
                0.0,  # 10: Lost Messages: Count
                0.0,  # 11: Output: Occupancy
                0.0,  # 12: Output: Oldest Age
                0.0,  # 13: Output: Youngest Age
                0.0,  # 14: Output: Average Age
            ]
        )
        state = np.array(system.state)
        np.testing.assert_allclose(expected_state, state)
        clock += 10

    # t == 50: Start compute task
    system.act([1])

    # t in [50, 60): Compute task running
    while clock.get_time() < 60:
        system.update()
        age = clock.get_time() / 100.0
        expected_state = np.array(
            [
                0.0,  #  0: Buffer 0: Occupancy
                0.0,  #  1: Buffer 0: Oldest Age
                0.0,  #  2: Buffer 0: Youngest Age
                0.0,  #  3: Buffer 0: Average Age
                0.0,  #  4: Buffer 1: Occupancy
                0.0,  #  5: Buffer 1: Oldest Age
                0.0,  #  6: Buffer 1: Youngest Age
                0.0,  #  7: Buffer 1: Average Age
                1.0,  #  8: Compute Node: Busy
                0.01 * (clock.get_time() - 50),  #  9: Compute Node: Age last started
                0.0,  # 10: Lost Messages: Count
                0.0,  # 11: Output: Occupancy
                0.0,  # 12: Output: Oldest Age
                0.0,  # 13: Output: Youngest Age
                0.0,  # 14: Output: Average Age
            ]
        )
        state = np.array(system.state)
        np.testing.assert_allclose(expected_state, state)
        clock += 1

    # t in [60, 100): Idle
    while clock.get_time() < 100:
        system.update()
        age = clock.get_time() / 100.0
        expected_state = np.array(
            [
                0.0,  #  0: Buffer 0: Occupancy
                0.0,  #  1: Buffer 0: Oldest Age
                0.0,  #  2: Buffer 0: Youngest Age
                0.0,  #  3: Buffer 0: Average Age
                0.0,  #  4: Buffer 1: Occupancy
                0.0,  #  5: Buffer 1: Oldest Age
                0.0,  #  6: Buffer 1: Youngest Age
                0.0,  #  7: Buffer 1: Average Age
                0.0,  #  8: Compute Node: Busy
                0.01 * (clock.get_time() - 50),  #  9: Compute Node: Age last started
                0.0,  # 10: Lost Messages: Count
                1.0,  # 11: Output: Occupancy
                0.01 * clock.get_time(),  # 12: Output: Oldest Age
                0.01 * clock.get_time(),  # 13: Output: Youngest Age
                0.01 * clock.get_time(),  # 14: Output: Average Age
            ]
        )
        state = np.array(system.state)
        np.testing.assert_allclose(expected_state, state)
        clock += 10
