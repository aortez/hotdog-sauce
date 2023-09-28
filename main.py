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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # torch_scalar = torch.tensor(3.14)
    # print(torch_scalar)
    #
    # x_np = np.random.random((4,4))
    # print("np: ", x_np)
    #
    # x_pt = torch.tensor(x_np)
    # print("pt: ", x_pt)
    #
    # print(x_np.dtype, x_pt.dtype)
    #
    # x_np = np.asarray(x_np, dtype=np.float32)
    # print("np:", x_np)
    #
    # x_pt = torch.tensor(x_np)
    # print("pt: ", x_pt)
    #
    # b_pt = (x_pt > 0.5)
    # print("b_pt: ", b_pt)
    #
    # print("np.sum(b_pt)", torch.sum(b_pt))
    #
    # x = torch.rand(2**11, 2**11)
    # time_cpu = timeit.timeit("x@x", globals=globals(), number=100)
    # print(time_cpu)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # print("device: ", device)
    #
    # x = x.to(device)
    #
    # time_gpu = timeit.timeit("x@x", globals=globals(), number=100)
    # print("time_gpu: ", time_gpu)
    #
    # print("speedup: ", time_cpu/time_gpu)
    #
    # some_tensors = [torch.tensor(1), torch.tensor(2)]
    # print("some_tensors: ", some_tensors)
    # print(moveTo(some_tensors, device))

    x_axis_vals = np.linspace(-7, 9, 100)
    y_axis_vals = f(torch.tensor(x_axis_vals)).numpy()

    print("x_axis_vals: ", x_axis_vals)
    print("y_axis_vals: ", y_axis_vals)
    #
    sns.lineplot(x=x_axis_vals, y=y_axis_vals, label = "f(x) = (x - 2)^2")
    plt.show()


