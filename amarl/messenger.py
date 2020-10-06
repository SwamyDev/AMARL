from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager

from amarl.metrics import PerformanceMeasure


class Message:
    TRAINING = 0
    ENV_TERMINATED = 1
    NUM_MESSAGES = 2


class _Messenger:
    def __init__(self):
        self._listeners = [list() for _ in range(Message.NUM_MESSAGES)]

    def subscribe_to(self, message, listener):
        self._listeners[message].append(listener)

    def unsubscribe_from(self, message, listener):
        self._listeners[message].remove(listener)

    def send(self, message, **kwargs):
        for listener in self._listeners[message]:
            listener(**kwargs)


_messenger = _Messenger()


@contextmanager
def subscription_to(message, listener):
    try:
        _messenger.subscribe_to(message, listener)
        yield
    finally:
        _messenger.unsubscribe_from(message, listener)


class ListeningMixin:
    def __init__(self):
        self._subscriptions = []

    def subscribe_to(self, message, listener):
        self._subscriptions.append((message, listener))
        _messenger.subscribe_to(message, listener)

    def close(self):
        for m, l in self._subscriptions:
            _messenger.unsubscribe_from(m, l)
        self._subscriptions.clear()

    def __del__(self):
        self.close()


class Monitor(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass


class LogMonitor(Monitor, ListeningMixin):
    TRAINING_LINE_FORMAT = "steps: {steps:>7}, average reward:\t{avg_reward:+.2f}"
    PERFORMANCE_LINE_FORMAT = "steps: {steps:>7}, performance:\t{performance:.2f} steps/s"

    def __init__(self, logger, progress_averaging, performance_sample_size):
        super().__init__()
        self._logger = logger
        self._progress_avg = progress_averaging
        self._performance_sample_size = performance_sample_size
        self._captured_returns = []
        self._mum_calls = 0
        self._last_return_logged = 0

        self._perf_measure = PerformanceMeasure()
        self._last_performance_logged = 0

    def start(self):
        self._perf_measure.start()
        self.subscribe_to(Message.TRAINING, self)

    def stop(self):
        self.close()

    @property
    def captured_returns(self):
        return self._captured_returns

    def __call__(self, **message):
        infos = message['infos']
        self._captured_returns.extend([i['episodic_return'] for i in infos if i.get('episodic_return')])
        self._mum_calls += len(infos)
        self._print_train_progress()
        self._print_wall_clock_performance()

    def _print_train_progress(self):
        if (len(self._captured_returns) - self._last_return_logged) >= self._progress_avg:
            begin = self._last_return_logged
            end = begin + self._progress_avg
            steps = self._mum_calls - (len(self._captured_returns) - end)
            avg = sum(self._captured_returns[begin:end]) / self._progress_avg
            self._log(self.TRAINING_LINE_FORMAT.format(steps=steps, avg_reward=avg))
            self._last_return_logged += self._progress_avg

    def _log(self, message):
        if self._logger:
            self._logger.info(message)

    def _print_wall_clock_performance(self):
        num_steps_elapsed = self._mum_calls - self._last_performance_logged
        if num_steps_elapsed >= self._performance_sample_size:
            self._perf_measure.stop()
            steps_per_sec = num_steps_elapsed / self._perf_measure.elapsed
            self._log(self.PERFORMANCE_LINE_FORMAT.format(steps=self._mum_calls, performance=steps_per_sec))
            self._last_performance_logged = self._mum_calls
            self._perf_measure.start()


class TensorboardMonitor(Monitor, ListeningMixin):
    def __init__(self, writer, progress_averaging, scalars):
        super().__init__()
        self._writer = writer
        self._window = deque(maxlen=progress_averaging)
        self._scalars = scalars
        self._step = 0

    @property
    def step(self):
        return self._step

    def start(self):
        self.subscribe_to(Message.TRAINING, self)

    def stop(self):
        self.close()

    def __call__(self, **message):
        infos = message['infos']
        for i in infos:
            for s in self._scalars:
                if s in i and i[s]:
                    self._window.append(i[s])
                    self._writer.add_scalar(self._scalars[s], sum(self._window) / len(self._window), self._step)
            self._step += 1


class CombinedMonitor(Monitor):
    def __init__(self, monitors):
        self._monitors = monitors

    @property
    def monitors(self):
        return self._monitors

    def start(self):
        for m in self._monitors:
            m.start()

    def stop(self):
        for m in self._monitors:
            m.stop()


@contextmanager
def monitor(m):
    try:
        m.start()
        yield m
    finally:
        m.stop()


def broadcast(message, **kwargs):
    _messenger.send(message, **kwargs)
