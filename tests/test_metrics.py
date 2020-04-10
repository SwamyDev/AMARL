import time

from pytest import approx

from amarl.metrics import calc_average, time_scope


def test_running_average_from_dicts():
    assert calc_average([dict(a=1, b=3), dict(a=1, b=-3)]) == dict(a=1, b=0)


def test_time_scope_measures_performance_of_scope():
    with time_scope() as measure:
        time.sleep(0.1)

    assert measure.elapsed == approx(0.1, abs=0.001)
