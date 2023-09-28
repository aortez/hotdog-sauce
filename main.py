# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timeit
from tqdm.autonotebook import tqdm
import torch

# def moveTo(obj, device):
#     """
#     obj: the python object to move to the device, or to move its contents to the device
#     device: the compute device to move the object to
#     """
#     if isinstance(obj, list):
#         return [moveTo(x, device) for x in obj]
#     elif isinstance(obj, tuple):
#         return tuple(moveTo(list(obj), device))
#     elif isinstance(obj, set):
#         return set(moveTo(list(obj), device))
#     elif isinstance(obj, dict):
#         to_ret = dict()
#         for key, value in obj.items():
#             to_ret[moveTo(key, device)] = moveTo(value, device)
#             return to_ret
#     elif hasattr(obj, "to"):
#         return obj.to(device)
#     else:
#         return obj


def f(x):
    return torch.pow((x - 2.0), 2)


def f_del(x):
    return 2*x-4


if __name__ == '__main__':
    x_axis_vals = np.linspace(-7, 9, 100)
    y_axis_vals = f(torch.tensor(x_axis_vals)).numpy()

    print("x_axis_vals: ", x_axis_vals)
    print("y_axis_vals: ", y_axis_vals)

    # Plot pretend loss function.
    sns.lineplot(x=x_axis_vals, y=y_axis_vals, label="f(x) = (x - 2)^2")

    # Draw line at known minimum.
    sns.lineplot(x=x_axis_vals, y=[0.0] * len(x_axis_vals), label="minimum", color='black')

    # Compute values of gradient of f(x).
    y_axis_vals_del = f_del(torch.tensor(x_axis_vals)).numpy()

    # Draw gradient of f(x).
    sns.lineplot(x=x_axis_vals, y=y_axis_vals_del, label="gradient of $f'(x) = 2x - 4$", color='red')

    # Pretend that we do not know the minimum, and we just look at a bunch of points along the derivative of the loss
    # function (f(x))). We can see that f(x) < 2 is negative and f(x) > 2 is positive. This tells us how to adjust
    # our guess at the minimum (which we already know is at x = 2).

    plt.show()
