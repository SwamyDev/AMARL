import gym
import numpy as np
import pytest
import torch

from amarl.processing import proc_screen, get_cart_pos_normalized, np_dtype_to_torch_dtype


def test_processing_image_data_to_fit_pytorch_model_dimensions():
    screen = np.random.randn(400, 600, 3)
    assert proc_screen(screen, 300).shape == (3, 40, 40)


def test_processing_image_crops_image_at_borders_correctly_when_player_is_moving_to_the_edge():
    screen = np.random.randn(400, 600, 3)
    assert proc_screen(screen, 0).shape == (3, 40, 40)
    assert proc_screen(screen, 600).shape == (3, 40, 40)


def test_get_normalized_position_of_cart_from_cart_pole_env():
    env = gym.make('CartPole-v0')
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


def test_casting_numpy_to_torch_dtype():
    assert np_dtype_to_torch_dtype(np.float) == torch.float64
    assert np_dtype_to_torch_dtype(np.float64) == torch.float64
    assert np_dtype_to_torch_dtype(np.float32) == torch.float32
    assert np_dtype_to_torch_dtype(np.float16) == torch.float16
    assert np_dtype_to_torch_dtype(np.long) == torch.long
    assert np_dtype_to_torch_dtype(np.int16) == torch.int16
    assert np_dtype_to_torch_dtype(np.short) == torch.int16
    assert np_dtype_to_torch_dtype(np.int64) == torch.int64
    assert np_dtype_to_torch_dtype(np.int) == torch.int64
    assert np_dtype_to_torch_dtype(np.uint8) == torch.uint8
    assert np_dtype_to_torch_dtype(np.int8) == torch.int8
    assert np_dtype_to_torch_dtype(np.char) == torch.int8
    assert np_dtype_to_torch_dtype(np.bool) == torch.bool


def test_raise_not_implemented_error_when_cast_is_not_implemented():
    with pytest.raises(NotImplementedError):
        np_dtype_to_torch_dtype(np.longlong)
