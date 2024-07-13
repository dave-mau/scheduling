import pytest

from .metrics import MovingAverage, MovingTotal


@pytest.fixture
def setup_running_total():
    return MovingTotal(2)


def test_running_total(setup_running_total):
    rt = setup_running_total
    assert rt.value == 0

    rt.push(1)
    assert rt.value == 1

    rt.push(2)
    assert rt.value == 3

    rt.push(10)
    assert rt.value == 12

    rt.push(-1)
    assert rt.value == 9


@pytest.fixture
def setup_running_avg():
    return MovingAverage(2)


def test_running_avg(setup_running_avg):
    rt = setup_running_avg
    assert pytest.approx(rt.value, 1.0e-6) == 0

    rt.push(1)
    assert pytest.approx(rt.value, 1.0e-6) == 1

    rt.push(2)
    assert pytest.approx(rt.value, 1.0e-6) == 1.5

    rt.push(10)
    assert pytest.approx(rt.value, 1.0e-6) == 6.0

    rt.push(-1)
    assert pytest.approx(rt.value, 1.0e-6) == 4.5
