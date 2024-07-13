import pytest
from computation_sim.basic_types import Header
from computation_sim.nodes import header_to_state


def test_header_to_state_pass():
    result = header_to_state(Header(1, 2, 3), 4)
    result[0] == pytest.approx(3.0, 1.0e-6)
    result[1] == pytest.approx(2.0, 1.0e-6)
    result[2] == pytest.approx(1.0, 1.0e-6)
