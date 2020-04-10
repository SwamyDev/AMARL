import time
from collections import defaultdict
from contextlib import contextmanager


def calc_average(last_rewards):
    avg = defaultdict(lambda: 0)
    for r in last_rewards:
        for i in r:
            avg[i] += r[i] / len(last_rewards)
    return avg


class PerformanceMeasure:
    def __init__(self):
        self._started = None
        self.elapsed = None

    def start(self):
        self._started = time.time()

    def stop(self):
        self.elapsed = time.time() - self._started


@contextmanager
def time_scope():
    m = PerformanceMeasure()
    try:
        m.start()
        yield m
    finally:
        m.stop()