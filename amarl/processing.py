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
