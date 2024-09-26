import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torchvision.datasets as dsets  # type: ignore
import torchvision.transforms as transforms  # type: ignore
from torchvision import transforms

# Set the seed for reproducibility
torch.manual_seed(0)

# Define the size of image
IMAGE_SIZE = 16


def show_data(data_sample: torch.Tensor) -> None:
    """Show image and label."""
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
    plt.title("y = " + str(data_sample[1]))


def compose_transforms() -> transforms.Compose:
    """Compose transforms."""
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    composed = transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    )
    return composed


def load_data(root: str = ".fashion/data") -> tuple:
    """Load Fashion MNIST dataset."""
    composed = compose_transforms()
    
    dataset_train = dsets.FashionMNIST(
        root=root, train=True, transform=composed, download=True
    )
    dataset_val = dsets.FashionMNIST(
        root=root, train=True, transform=composed, download=True
    )
    return dataset_train, dataset_val


def show_sample_screenshots() -> None:
    """Show 3 sample screenshots."""
    _, dataset_val = load_data()

    for n, data_sample in enumerate(dataset_val):

        show_data(data_sample)
        plt.show()
        if n == 2:
            break


class CNN_batch(nn.Module):
    """Convolutional Neural Network with Batch Normalization."""

    def __init__(self, out_1=16, out_2=32, number_of_classes=10):
        super(CNN_batch, self).__init__()
        self.cnn1 = nn.Conv2d(
            in_channels=1, out_channels=out_1, kernel_size=5, padding=2
        )
        self.conv1_bn = nn.BatchNorm2d(out_1)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(
            in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2
        )
        self.conv2_bn = nn.BatchNorm2d(out_2)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2 * 4 * 4, number_of_classes)
        self.bn_fc1 = nn.BatchNorm1d(10)

    def forward(self, x):
        """Forward propagation."""
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        return x


class CNN(nn.Module):
    """Convolutional Neural Network without Batch Normalization."""

    def __init__(self, out_1=16, out_2=32, number_of_classes=10):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(
            in_channels=1, out_channels=out_1, kernel_size=5, padding=2
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(
            in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2 * 4 * 4, number_of_classes)

    def forward(self, x):
        """Forward propagation."""
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def train_model() -> tuple:
    """Train model."""
    dataset_train, dataset_val = load_data()

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=100)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=100)

    # Define the model architecture
    model = CNN_batch(out_1=16, out_2=32, number_of_classes=10)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    cost_list = []
    accuracy_list = []
    N_test = len(dataset_val)
    n_epochs = 5

    for _ in range(n_epochs):
        cost = 0
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            cost += loss.item()

        correct = 0

        # Perform a prediction on the validation data
        model.eval()

        for x_test, y_test in test_loader:
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
        cost_list.append(cost)
    return cost_list, accuracy_list


def plot_results(cost_list: list, accuracy_list: list) -> None:
    """Plot results."""
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.plot(cost_list, color=color)
    ax1.set_xlabel("epoch", color=color)
    ax1.set_ylabel("Cost", color=color)
    ax1.tick_params(axis="y", color=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("accuracy", color=color)
    ax2.set_xlabel("epoch", color=color)
    ax2.plot(accuracy_list, color=color)
    ax2.tick_params(axis="y", color=color)
    fig.tight_layout()


if __name__ == "__main__":
    # First assignment
    show_sample_screenshots()

    # Second assignment
    cost_list, accuracy_list = train_model()
    plot_results(cost_list, accuracy_list)
