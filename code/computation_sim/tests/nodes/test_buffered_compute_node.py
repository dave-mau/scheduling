import pytest
from unittest.mock import Mock

from computation_sim.nodes import Node, BufferedComputeNode
from computation_sim.basic_types import (
    BadNodeGraphError,
    Message,
    Header,
    CommunicationError,
)


@pytest.fixture
def compute_node_ok():
    mock_compute = Mock(spec=Node)
    mock_output = Mock(spec=Node)
    mock_compute.outputs = [mock_output]
    mock_buf_0 = Mock(spec=Node)
    mock_buf_0.outputs = [mock_compute]
    mock_buf_1 = Mock(spec=Node)
    mock_buf_1.outputs = [mock_compute]

    result = BufferedComputeNode(Mock(), mock_compute)

    result.set_buffer_for_sender("sender_0", mock_buf_0)
    result.set_buffer_for_sender("sender_1", mock_buf_1)

    mocks = {
        "mock_compute": mock_compute,
        "mock_buf_0": mock_buf_0,
        "mock_buf_1": mock_buf_1,
        "mock_output": mock_output,
    }
    return mocks, result


def test_outputs(compute_node_ok):
    mocks, node = compute_node_ok
    result = node.outputs

    assert len(result) == 1
    assert result[0] == mocks["mock_output"]


def test_set_buffer_output_not_set():
    mock_compute = Mock(spec=Node)
    mock_buf_0 = Mock(spec=Node)
    mock_buf_0.outputs = []

    node = BufferedComputeNode(Mock(), mock_compute)

    with pytest.raises(BadNodeGraphError):
        node.set_buffer_for_sender("sender_0", mock_buf_0)


def test_receive_pass(compute_node_ok):
    mocks, node = compute_node_ok

    header = Header()
    header.sender_id = "sender_1"
    node.receive(Message(header))

    mocks["mock_buf_0"].receive.assert_not_called()
    mocks["mock_buf_1"].receive.assert_called()


def test_receive_fail(compute_node_ok):
    mocks, node = compute_node_ok

    header = Header()
    header.sender_id = "sender_unknown"

    with pytest.raises(CommunicationError):
        node.receive(Message(header))


def test_state(compute_node_ok):
    mocks, node = compute_node_ok

    mocks["mock_buf_0"].state = [1, 2]
    mocks["mock_buf_1"].state = [3, 4]
    mocks["mock_compute"].state = [5, 6]
    mocks["mock_output"].state = [7, 8]

    result = node.state

    assert result == [1, 2, 3, 4, 5, 6]
