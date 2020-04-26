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


def subscribe_to(message, listener):
    _messenger.subscribe_to(message, listener)


def unsubscribe_from(message, listener):
    _messenger.unsubscribe_from(message, listener)


@contextmanager
def subscription_to(message, listener):
    try:
        _messenger.subscribe_to(message, listener)
        yield
    finally:
        _messenger.unsubscribe_from(message, listener)


class TrainingMonitor:
    TRAINING_LINE_FORMAT = "steps: {steps:>7}, average reward:\t{avg_reward:+.2f}"
    PERFORMANCE_LINE_FORMAT = "steps: {steps:>7}, performance:\t{performance:.2f} steps/s"

    def __init__(self, logger, progress_averaging, performance_sample_size):
        self._logger = logger
        self._progress_avg = progress_averaging
        self._performance_sample_size = performance_sample_size
        self._captured_returns = []
        self._mum_calls = 0
        self._last_return_logged = 0

        self._perf_measure = PerformanceMeasure()
        self._perf_measure.start()
        self._last_performance_logged = 0

        _messenger.subscribe_to(Message.TRAINING, self)

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

    def close(self):
        _messenger.unsubscribe_from(Message.TRAINING, self)


@contextmanager
def training_monitor(logger=None, progress_averaging=100, performance_sample_size=1000):
    m = TrainingMonitor(logger, progress_averaging, performance_sample_size)
    try:
        yield m
    finally:
        m.close()


def broadcast(message, **kwargs):
    _messenger.send(message, **kwargs)
