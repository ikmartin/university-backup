# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # relu, tanh
from torch.utils.data import DataLoader  # easier dataset management
import torchvision.datasets as datasets  #
import torchvision.transforms as transforms


# neural net for problem 4a
class NN4A(nn.Module):
    # this initializes all the neural net layers
    def __init__(self, num_classes):
        super(NN4A, self).__init__()
        # the convolution
        self.conv1 = nn.LazyConv2d(
            out_channels=20,
            kernel_size=(3, 3),
            stride=(3, 3),
            padding=(0, 0),
        )  # this is called a "same convolution"

        # this is the pooling
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # this turns a shape (64, 20, 4, 4) tensor into a shape (64, 320) tensor
        self.flatten = nn.Flatten(start_dim=1)
        # fully connected layer 1
        self.fc1 = nn.LazyLinear(100)
        # fully connected layer 2
        self.fc2 = nn.LazyLinear(num_classes)
        # this is just so we can see the shape of the vector through one pass of the network
        self._num = 0

        # NOTE
        # WE DON'T DO A SOFTMAX AT THE END SINCE IT'S INCLUDED IN OUR CHOICE OF LOSS FUNCTION

    # this runs a data point x through the neural net once
    def forward(self, x):
        # print one pass through the network
        if self._num == 0:
            self.shapeprint(x)
            self._num += 1

        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def shapeprint(self, x):
        x = F.leaky_relu(self.conv1(x))
        print("shape after conv:", x.shape)
        x = self.pool(x)
        print("shape after pool:", x.shape)
        x = self.flatten(x)
        print("shape after pool:", x.shape)
        x = F.leaky_relu(self.fc1(x))
        print("shape after fc1:", x.shape)
        x = self.fc2(x)
        print("shape after fc2:", x.shape)


# neural net for problem 4a
class NN4B(nn.Module):
    # this initializes all the neural net layers
    def __init__(self, input_size, num_classes):
        super(NN4B, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.LazyLinear(100)
        self.fc3 = nn.LazyLinear(num_classes)

        # NOTE
        # WE DON'T DO A SOFTMAX AT THE END SINCE IT'S INCLUDED IN OUR CHOICE OF LOSS FUNCTION

    # this runs a data point x through the neural net once
    def forward(self, x):
        x = self.flatten(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
inchannels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Load Data

# all we use transform for is to convert data to tensor
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize Network
model_4a = NN4A(num_classes=num_classes).to(device)
model_4b = NN4B(input_size=784, num_classes=num_classes).to(device)

# Loss and optimizer 4A
criterion_4a = nn.CrossEntropyLoss()
optimizer_4a = optim.Adam(model_4a.parameters(), lr=learning_rate)

# Loss and optimizer 4B
criterion_4b = nn.CrossEntropyLoss()
optimizer_4b = optim.Adam(model_4b.parameters(), lr=learning_rate)

# Train Network

print("Device used: {}".format(device))


def train4a():
    print("\nTRAINING 4A\n-------------\n")
    for epoch in range(num_epochs):
        print("4A EPOCH: ", epoch)
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores4a = model_4a(data)
            loss4a = criterion_4a(scores4a, targets)

            # backward
            optimizer_4a.zero_grad()
            loss4a.backward()

            # gradient descent or adam step
            optimizer_4a.step()  # update the weights depending on loss computed in loss.backward()


# train 4B
def train4b():
    print("TRAINING 4B")
    for epoch in range(num_epochs):
        print("4B EPOCH: ", epoch)
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores4b = model_4b(data)
            loss4b = criterion_4b(scores4b, targets)

            # backward
            optimizer_4b.zero_grad()
            loss4b.backward()

            # gradient descent or adam step
            optimizer_4b.step()  # update the weights depending on loss computed in loss.backward()


# Check accuracy on training and test our Network
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)

            # 64 x 10
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) * 100:.2f}"
        )
    model.train()


train4a()
train4b()

print("###########################\n## PROBLEM 4A\n###########################")
check_accuracy(train_loader, model_4a)
check_accuracy(test_loader, model_4a)
print("\n\n###########################\n## PROBLEM 4B\n###########################")
check_accuracy(train_loader, model_4b)
check_accuracy(test_loader, model_4b)
