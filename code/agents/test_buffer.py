import pytest

from .buffer import Memory, Sample


@pytest.fixture
def setup():
    memory = Memory(5)
    memory.push(1, 2, 3, 4)
    memory.push(11, 12, 13, 14)
    memory.push(21, 22, 23, 24)
    memory.push(31, 32, 33, 34)
    memory.push(41, 42, 43, 44)
    memory.push(51, 52, 53, 54)
    return memory


def test_push_circular(setup):
    assert Sample(1, 2, 3, 4) not in setup
    assert Sample(11, 12, 13, 14) in setup
    assert Sample(21, 22, 23, 24) in setup
    assert Sample(31, 32, 33, 34) in setup
    assert Sample(41, 42, 43, 44) in setup
    assert Sample(51, 52, 53, 54) in setup
    assert len(setup) == 5


def test_push_non_circular():
    memory = Memory(5)
    memory.push(1, 2, 3, 4)
    assert Sample(1, 2, 3, 4) in memory
    assert len(memory) == 1


def test_sample(setup):
    result = setup.sample(2)
    assert len(result) == 2
    assert result[0] in setup
    assert result[1] in setup
