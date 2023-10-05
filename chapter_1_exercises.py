import matplotlib.pyplot as pyplot
import seaborn as sns
import sklearn
import timeit
import torch
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.datasets import fetch_openml
import utils as utils

torch_tensor3d = torch.tensor([
    [
        [1, 2, 3],
        [4, 5, 6],
    ],
    [
        [7, 8, 9],
        [10, 11, 12],
    ],
    [
        [13, 14, 15],
        [16, 17, 18],
    ],
    [
        [19, 20, 21],
        [22, 23, 24],
    ]
])


# Use for loops to compute average for torch_tensor3d.
def exercise1():
    total = 0
    for i in range(torch_tensor3d.shape[0]):
        for j in range(torch_tensor3d.shape[1]):
            for k in range(torch_tensor3d.shape[2]):
                total += torch_tensor3d[i, j, k]

    actual_avg = total / torch_tensor3d.numel()
    expected_avg = torch_tensor3d.mean(dtype=torch.float32)
    print("avg: ", actual_avg)
    assert \
        actual_avg == expected_avg, \
        f"actual_avg({actual_avg}) != expected_avg({expected_avg})"


# Print out the element with the value 13.
def exercise2():
    print("should be 13: ", torch_tensor3d[2, 0, 0])


# For every power of 2 from 0 to 11, create a random 2d matrix of size 2^n x 2^n and time how long it takes to compute
# the square of the matrix.  Compute on both the CPU and the GPU and plot the speedup.
def exercise3():
    assert torch.cuda.is_available(), "CUDA is not available."
    device = torch.device("cuda")

    x_axis_vals = []
    y_axis_vals_cpu = []
    y_axis_vals_gpu = []
    for n in range(12):
        x_axis_vals.append(n)

        size = 2 ** n
        x = torch.rand(size, size)
        globals()["x"] = x

        time_cpu = timeit.timeit("x@x", globals=globals(), number=100)
        utils.move_to(x, device)
        time_gpu = timeit.timeit("x@x", globals=globals(), number=100)
        y_axis_vals_cpu.append(time_cpu)
        y_axis_vals_gpu.append(time_gpu)

        print(f"n: {n}, size: {size}, time_cpu: {time_cpu}, time_gpu: {time_gpu}")

    sns.lineplot(x=x_axis_vals, y=y_axis_vals_cpu, label="CPU")
    sns.lineplot(x=x_axis_vals, y=y_axis_vals_gpu, label="GPU")
    pyplot.show()


class SimpleDataset(Dataset):

    def __init__(self, inputs, targets):
        super(SimpleDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, index):
        i = torch.tensor(self.inputs.iloc[index, :], dtype=torch.float32)
        t = torch.tensor(int(self.targets[index]), dtype=torch.int64)
        return i, t

    def __len__(self):
        return self.inputs.shape[0]


# So far just loads the dataset and shows some of the training data.
def use_mnist784():
    training_inputs, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
    print("trainingData.shape:", training_inputs.shape)
    print("labels.shape:", labels.shape)

    dataset = SimpleDataset(training_inputs, labels)
    print("dataset.len: ", len(dataset))

    for row in range(10):
        show_image_and_label(dataset[row][0], dataset[row][1])


def show_image_and_label(example, label):
    pyplot.imshow(example.reshape(28, 28), cmap='gray')
    pyplot.title("Label: " + str(label))
    pyplot.show()


def do_exercises():
    exercise1()
    exercise2()
    exercise3()


if __name__ == '__main__':
    do_exercises()
