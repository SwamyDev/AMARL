import numpy as np


def assert_obs_eq(actual, expected):
    np.testing.assert_equal(actual, expected)


def assert_selective_obs_eq(actual, expected):
    assert set(actual.keys()) == set(expected.keys())
    for k in actual:
        assert_obs_eq(actual[k], expected[k])
