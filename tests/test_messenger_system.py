import logging
import time

from amarl.messenger import training_monitor, broadcast, Message
from amarl.metrics import time_scope

logger = logging.getLogger(__name__)


def test_training_monitor_receives_and_logs_training_message(caplog):
    with caplog.at_level(logging.INFO):
        with training_monitor(logger=logger, progress_averaging=4) as m:
            broadcast(Message.TRAINING, infos=[{'episodic_return': 2}] * 4)

    assert caplog.messages == [m.TRAINING_LINE_FORMAT.format(steps=4, avg_reward=2.0)]


def test_training_monitor_receives_and_aggregates_multiple_messages(caplog):
    with caplog.at_level(logging.INFO):
        with training_monitor(logger=logger, progress_averaging=3) as m:
            broadcast(Message.TRAINING, infos=[{'episodic_return': 2}] * 1)
            broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}] * 2)

    assert caplog.messages == [m.TRAINING_LINE_FORMAT.format(steps=3, avg_reward=1 / 3)]


def test_training_monitor_can_print_multiple_monitor_messages(caplog):
    with caplog.at_level(logging.INFO):
        with training_monitor(logger=logger, progress_averaging=1) as m:
            broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])
            broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}])

    assert caplog.messages == [m.TRAINING_LINE_FORMAT.format(steps=1, avg_reward=2.0),
                               m.TRAINING_LINE_FORMAT.format(steps=2, avg_reward=-0.5)]


def test_training_monitor_prints_training_monitor_messages_at_specified_frequency_even_at_uneven_intervals(caplog):
    with caplog.at_level(logging.INFO):
        with training_monitor(logger=logger, progress_averaging=4) as m:
            broadcast(Message.TRAINING, infos=[{'episodic_return': +2}] * 3)
            broadcast(Message.TRAINING, infos=[{'episodic_return': -1}] * 2)
            broadcast(Message.TRAINING, infos=[{'episodic_return': +1}] * 3)

    assert caplog.messages == [m.TRAINING_LINE_FORMAT.format(steps=4, avg_reward=5 / 4),
                               m.TRAINING_LINE_FORMAT.format(steps=8, avg_reward=2 / 4)]


def test_training_monitor_ignores_episodic_return_being_none(caplog):
    with caplog.at_level(logging.INFO):
        with training_monitor(logger=logger, progress_averaging=2) as m:
            broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}])
            broadcast(Message.TRAINING, infos=[{'episodic_return': None}])
            assert not caplog.messages
            broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])

    assert caplog.messages == [m.TRAINING_LINE_FORMAT.format(steps=3, avg_reward=0.75)]


def test_training_monitor_ignores_episodic_return_being_not_present(caplog):
    with caplog.at_level(logging.INFO):
        with training_monitor(logger=logger, progress_averaging=2) as m:
            broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}])
            broadcast(Message.TRAINING, infos=[{}])
            assert not caplog.messages
            broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])
            assert caplog.messages == [m.TRAINING_LINE_FORMAT.format(steps=3, avg_reward=0.75)]


def test_training_monitor_prints_wall_clock_performance_in_specified_interval(caplog):
    with caplog.at_level(logging.INFO):
        with time_scope() as measure:
            with training_monitor(logger=logger, performance_sample_size=2) as m:
                time.sleep(0.2)
                broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])
                time.sleep(0.2)
                broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}])

    eps = 1e-5
    assert caplog.messages == [m.PERFORMANCE_LINE_FORMAT.format(steps=2, performance=2 / measure.elapsed)] or \
           caplog.messages == [m.PERFORMANCE_LINE_FORMAT.format(steps=2, performance=2 / (measure.elapsed + eps))] or \
           caplog.messages == [m.PERFORMANCE_LINE_FORMAT.format(steps=2, performance=2 / measure.elapsed - eps)]


def test_training_monitor_prints_multiple_wall_clock_performance_measures(caplog):
    with caplog.at_level(logging.INFO):
        with time_scope() as measure:
            with training_monitor(logger=logger, performance_sample_size=2) as m:
                for _ in range(4):
                    time.sleep(0.2)
                    broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])

    assert caplog.messages == [m.PERFORMANCE_LINE_FORMAT.format(steps=2, performance=4 / measure.elapsed),
                               m.PERFORMANCE_LINE_FORMAT.format(steps=4, performance=4 / measure.elapsed)]


def test_training_monitor_prints_wall_clock_performance_measure_at_specified_frequency_even_at_uneven_calls(caplog):
    with caplog.at_level(logging.INFO):
        with time_scope() as measure:
            with training_monitor(logger=logger, performance_sample_size=4) as m:
                time.sleep(0.5)
                broadcast(Message.TRAINING, infos=[{'episodic_return': 2}] * 3)
                time.sleep(0.3)
                broadcast(Message.TRAINING, infos=[{'episodic_return': 2}] * 2)

    assert caplog.messages == [m.PERFORMANCE_LINE_FORMAT.format(steps=5, performance=5 / measure.elapsed)]


def test_training_monitor_captures_rewards():
    with training_monitor() as monitor:
        broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])
        broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}])

    assert monitor.captured_returns == [2, -0.5]


def test_training_monitor_logger_is_optional(caplog):
    with caplog.at_level(logging.INFO):
        with training_monitor(progress_averaging=1, performance_sample_size=2):
            broadcast(Message.TRAINING, infos=[{'episodic_return': 2}])
            broadcast(Message.TRAINING, infos=[{'episodic_return': -0.5}])

    assert not caplog.messages
