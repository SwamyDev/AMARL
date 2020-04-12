import pytest


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
