import pytest
from computation_sim.basic_types import Header
from computation_sim.nodes import header_to_state, empty_message_state


def test_header_to_state_pass():
    result = header_to_state(Header(1, 2, 3, 11), 4)
    assert len(result) == 4
    result[0] == pytest.approx(3.0, 1.0e-6)
    result[1] == pytest.approx(2.0, 1.0e-6)
    result[2] == pytest.approx(1.0, 1.0e-6)
    result[3] == pytest.approx(11.0, 1.0e-6)

def test_empty_message_state_pass():
    result = empty_message_state()
    assert len(result) == 4
    result[0] == pytest.approx(0.0, 1.0e-6)
    result[1] == pytest.approx(0.0, 1.0e-6)
    result[2] == pytest.approx(0.0, 1.0e-6)
    result[3] == pytest.approx(0.0, 1.0e-6)
