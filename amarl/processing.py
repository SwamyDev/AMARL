import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def calc_horizontal_slice(cart_pos, screen_width, screen_height):
    view_width = int(screen_height * 0.4)
    if cart_pos < view_width // 2:
        return slice(view_width)
    elif cart_pos > (screen_width - view_width // 2):
        return slice(-view_width, None)
    else:
        return slice(cart_pos - view_width // 2, cart_pos + view_width // 2)


def proc_screen(screen, player_pos):
    screen = screen.transpose((2, 0, 1))
    _, h, w = screen.shape
    screen = screen[:, int(h * 0.4):int(h * 0.8)]
    w_slice = calc_horizontal_slice(player_pos, w, h)
    screen = screen[:, :, w_slice]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    return resize(torch.from_numpy(screen))


def get_cart_pos_normalized(env, width_px):
    while env.unwrapped is not None and not hasattr(env, 'x_threshold'):
        env = env.unwrapped

    world_width = env.x_threshold * 2
    scale = width_px / world_width
    return int(env.state[0] * scale + width_px / 2.0)


def np_dtype_to_torch_dtype(np_dtype):
    if np_dtype == np.float or np_dtype == np.float64:
        return torch.float64
    if np_dtype == np.float32:
        return torch.float32
    if np_dtype == np.float16:
        return torch.float16
    if np_dtype == np.long:
        return torch.long
    if np_dtype == np.int64:
        return torch.int64
    if np_dtype == np.int:
        return torch.int
    if np_dtype == np.int32:
        return torch.int32
    if np_dtype == np.int16:
        return torch.int16
    if np_dtype == np.int8 or np_dtype == np.char:
        return torch.int8
    if np_dtype == np.uint8:
        return torch.uint8
    if np_dtype == np.bool:
        return torch.bool

    raise NotImplementedError(f"cast from {np_dtype} not implemented.")
