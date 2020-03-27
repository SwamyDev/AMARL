import gym
import numpy as np

from amarl.processing import proc_screen, get_cart_pos_normalized


def test_processing_image_data_to_fit_pytorch_model_dimensions():
    screen = np.random.randn(400, 600, 3)
    assert proc_screen(screen, 300).shape == (3, 40, 40)


def test_processing_image_crops_image_at_borders_correctly_when_player_is_moving_to_the_edge():
    screen = np.random.randn(400, 600, 3)
    assert proc_screen(screen, 0).shape == (3, 40, 40)
    assert proc_screen(screen, 600).shape == (3, 40, 40)


def test_get_normalized_position_of_cart_from_cart_pole_env():
    env = gym.make('CartPole-v0').unwrapped
    env.seed(42)
    env.reset()
    assert get_cart_pos_normalized(env, 600) == 298
    repeat_action(env, 0, 2)
    assert get_cart_pos_normalized(env, 600) == 297
    repeat_action(env, 1, 6)
    assert get_cart_pos_normalized(env, 600) == 299


def repeat_action(env, action, n):
    for _ in range(n):
        env.step(action)
