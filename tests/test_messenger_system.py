import logging
import time

from amarl.messenger import monitor, broadcast, Message, LogMonitor, ListeningMixin, CombinedMonitor, Monitor, \
    TensorboardMonitor
from amarl.metrics import time_scope

logger = logging.getLogger(__name__)


def make_training_monitor(logger=None, progress_averaging=100, performance_sample_size=1000):
    return LogMonitor(logger, progress_averaging, performance_sample_size)


def test_training_monitor_receives_and_logs_training_message(caplog):
    with caplog.at_level(logging.INFO):
        with monitor(make_training_monitor(logger=logger, progress_averaging=4)) as m:
            broadcast(Message.TRAINING, infos=[{'episodic_return': 2}] * 4)

    assert caplog.messages == [m.TRAINING_LINE_FORMAT.format(steps=4, avg_reward=2.0)]


def test_training_monitor_receives_and_aggregates_multiple_messages(caplog):
    with caplog.at_level(logging.INFO):
        with monitor(make_training_monitor(logger=logger, progress_averaging=3)) as m:
            broadcast(Message.TRAINING, infos=[{'episodic_return': 2}] * 1)
            broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}] * 2)

    assert caplog.messages == [m.TRAINING_LINE_FORMAT.format(steps=3, avg_reward=1 / 3)]


def test_training_monitor_can_print_multiple_monitor_messages(caplog):
    with caplog.at_level(logging.INFO):
        with monitor(make_training_monitor(logger=logger, progress_averaging=1)) as m:
            broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])
            broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}])

    assert caplog.messages == [m.TRAINING_LINE_FORMAT.format(steps=1, avg_reward=2.0),
                               m.TRAINING_LINE_FORMAT.format(steps=2, avg_reward=-0.5)]


def test_training_monitor_prints_training_monitor_messages_at_specified_frequency_even_at_uneven_intervals(caplog):
    with caplog.at_level(logging.INFO):
        with monitor(make_training_monitor(logger=logger, progress_averaging=4)) as m:
            broadcast(Message.TRAINING, infos=[{'episodic_return': +2}] * 3)
            broadcast(Message.TRAINING, infos=[{'episodic_return': -1}] * 2)
            broadcast(Message.TRAINING, infos=[{'episodic_return': +1}] * 3)

    assert caplog.messages == [m.TRAINING_LINE_FORMAT.format(steps=4, avg_reward=5 / 4),
                               m.TRAINING_LINE_FORMAT.format(steps=8, avg_reward=2 / 4)]


def test_training_monitor_ignores_episodic_return_being_none(caplog):
    with caplog.at_level(logging.INFO):
        with monitor(make_training_monitor(logger=logger, progress_averaging=2)) as m:
            broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}])
            broadcast(Message.TRAINING, infos=[{'episodic_return': None}])
            assert not caplog.messages
            broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])

    assert caplog.messages == [m.TRAINING_LINE_FORMAT.format(steps=3, avg_reward=0.75)]


def test_training_monitor_ignores_episodic_return_being_not_present(caplog):
    with caplog.at_level(logging.INFO):
        with monitor(make_training_monitor(logger=logger, progress_averaging=2)) as m:
            broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}])
            broadcast(Message.TRAINING, infos=[{}])
            assert not caplog.messages
            broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])
            assert caplog.messages == [m.TRAINING_LINE_FORMAT.format(steps=3, avg_reward=0.75)]


def test_training_monitor_prints_wall_clock_performance_in_specified_interval(caplog):
    with caplog.at_level(logging.INFO):
        with time_scope() as measure:
            with monitor(make_training_monitor(logger=logger, performance_sample_size=2)) as m:
                time.sleep(0.2)
                broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])
                time.sleep(0.2)
                broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}])

    eps = 0.01
    assert caplog.messages == [m.PERFORMANCE_LINE_FORMAT.format(steps=2, performance=2 / measure.elapsed)] or \
           caplog.messages == [m.PERFORMANCE_LINE_FORMAT.format(steps=2, performance=2 / measure.elapsed + eps)] or \
           caplog.messages == [m.PERFORMANCE_LINE_FORMAT.format(steps=2, performance=2 / measure.elapsed - eps)]


def test_training_monitor_prints_multiple_wall_clock_performance_measures(caplog):
    with caplog.at_level(logging.INFO):
        with time_scope() as measure:
            with monitor(make_training_monitor(logger=logger, performance_sample_size=2)) as m:
                for _ in range(4):
                    time.sleep(0.2)
                    broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])

    eps = 0.01
    assert caplog.messages == [m.PERFORMANCE_LINE_FORMAT.format(steps=2, performance=4 / measure.elapsed),
                               m.PERFORMANCE_LINE_FORMAT.format(steps=4, performance=4 / measure.elapsed)] or \
           caplog.messages == [m.PERFORMANCE_LINE_FORMAT.format(steps=2, performance=4 / measure.elapsed + eps),
                               m.PERFORMANCE_LINE_FORMAT.format(steps=4, performance=4 / measure.elapsed + eps)] or \
           caplog.messages == [m.PERFORMANCE_LINE_FORMAT.format(steps=2, performance=4 / measure.elapsed - eps),
                               m.PERFORMANCE_LINE_FORMAT.format(steps=4, performance=4 / measure.elapsed - eps)]


def test_training_monitor_prints_wall_clock_performance_measure_at_specified_frequency_even_at_uneven_calls(caplog):
    with caplog.at_level(logging.INFO):
        with time_scope() as measure:
            with monitor(make_training_monitor(logger=logger, performance_sample_size=4)) as m:
                time.sleep(0.5)
                broadcast(Message.TRAINING, infos=[{'episodic_return': 2}] * 3)
                time.sleep(0.3)
                broadcast(Message.TRAINING, infos=[{'episodic_return': 2}] * 2)

    eps = 0.01
    assert caplog.messages == [m.PERFORMANCE_LINE_FORMAT.format(steps=5, performance=5 / measure.elapsed)] or \
           caplog.messages == [m.PERFORMANCE_LINE_FORMAT.format(steps=5, performance=5 / measure.elapsed + eps)] or \
           caplog.messages == [m.PERFORMANCE_LINE_FORMAT.format(steps=5, performance=5 / measure.elapsed - eps)]


def test_training_monitor_captures_rewards():
    with monitor(make_training_monitor()) as m:
        broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])
        broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}])

    assert m.captured_returns == [2, -0.5]


def test_training_monitor_logger_is_optional(caplog):
    with caplog.at_level(logging.INFO):
        with monitor(make_training_monitor(progress_averaging=1, performance_sample_size=2)):
            broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])
            broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}])

    assert not caplog.messages


class MonitorSpy(Monitor, ListeningMixin):
    def __init__(self):
        super().__init__()
        self.num_messages_received = 0

    def start(self):
        self.subscribe_to(Message.TRAINING, self)

    def stop(self):
        self.close()

    def __call__(self, **message):
        self.num_messages_received += 1


def test_combined_monitors():
    with monitor(CombinedMonitor([MonitorSpy(), MonitorSpy()])) as m:
        broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])
        broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}])
    broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}])

    assert m.monitors[0].num_messages_received == 2
    assert m.monitors[1].num_messages_received == 2


class SummaryWriterSpy:
    def __init__(self):
        self.received_scalars = []

    def add_scalar(self, path, value, step):
        self.received_scalars.append((path, value, step))


def test_tensorboard_monitor_writes_specified_infos():
    writer = SummaryWriterSpy()
    tb_monitor = TensorboardMonitor(writer, progress_averaging=2, scalars=dict(episodic_return='returns/episodic'))
    with monitor(tb_monitor):
        broadcast(Message.TRAINING, infos=[{'episodic_return': 2}] * 2)
        broadcast(Message.TRAINING, infos=[{'episodic_return': None}])
        broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5, 'other_value': "its a string"}])
    broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])

    assert writer.received_scalars == [
        ('returns/episodic', 2, 0),
        ('returns/episodic', 2, 1),
        ('returns/episodic', 0.75, 3)
    ]
    assert tb_monitor.step == 4
