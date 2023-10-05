import torch
import matplotlib.pyplot
import numpy as np
import pandas as pd

def f(x):
    return torch.pow((x - 2.0), 2)


def f_del(x):
    return 2*x-4


def move_to(obj, device):
    """
    obj: the python object to move to the device, or to move its contents to the device
    device: the compute device to move the object to
    """
    if isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
            return to_ret
    elif hasattr(obj, "to"):
        return obj.to(device)
    else:
        return obj

def do_it():
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

    # plt.show()

    x = torch.tensor(2.0, requires_grad=True)
    print("x.grad: ", x.grad)

    value = f(x)
    print("value: ", value)

    value.backward()
    print("x.grad: ", x.grad)

    x = torch.tensor([-3.5], requires_grad=True)

    x_cur = x.clone()
    x_prev = x_cur * 100

    epsilon = 1e-5

    eta = 0.1

    while torch.linalg.norm(x_cur - x_prev) > epsilon:
        x_prev = x_cur.clone()

        value = f(x)
        value.backward()
        x.data -= eta * x.grad

        x.grad.zero_()

        x_cur = x.data

    print("x_cur: ", x_cur)

    x_param = torch.nn.Parameter(torch.tensor([-3.5]), requires_grad=True)

    optimizer = torch.optim.SGD([x_param], lr=eta)

    for epoch in range(60):
        optimizer.zero_grad()
        loss_incurred = f(x_param)
        loss_incurred.backward()
        optimizer.step()

    print("x_param: ", x_param)
