from amarl.main import calc_average


def test_running_average_from_dicts():
    assert calc_average([dict(a=1, b=3), dict(a=1, b=-3)]) == dict(a=1, b=0)
