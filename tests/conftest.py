import pytest

from amarl.wrappers import MultipleEnvs
from tests.doubles.envs import EnvSpy


def pytest_addoption(parser):
    parser.addoption(
        "--run-rendered", action="store_true", default=False, help="run tests using renderer (OpenGL)"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "rendered: mark test as using rendering (OpenGL) features"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-rendered"):
        return

    skip_rendered = pytest.mark.skip(reason="need --run-rendered to run")
    for item in items:
        if "rendered" in item.keywords:
            item.add_marker(skip_rendered)


@pytest.fixture
def recorded_close_calls():
    return list()


@pytest.fixture
def recorded_render_calls():
    return list()


@pytest.fixture
def multiple_envs_selective(obs_space, default_obs, recorded_close_calls, recorded_render_calls):
    return MultipleEnvs(lambda: EnvSpy(obs_space, default_obs, recorded_close_calls, recorded_render_calls), 5,
                        is_selective=True)
