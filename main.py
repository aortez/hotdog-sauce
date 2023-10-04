import matplotlib.pyplot as pyplot
import seaborn as sns
import sklearn
import timeit
import torch
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.datasets import fetch_openml


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


def main() -> None:

    # Load MNIST dataset.
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


if __name__ == '__main__':
    main()
