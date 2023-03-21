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
    def __init__(self, num_classes):
        super(NN4A, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=20,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # this is called a same convolution
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = self.pool(x)
        print(x.shape)
        # flattens to a two coordinate thing
        x = x.reshape(x.shape[0], -1)
        print(x.shape)
        x = self.fc1(x)

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
model = NN4A().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network

print("Device used: {}".format(device))
# one epoch means network has seen all images in training set
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()  # update the weights depending on loss computed in loss.backward()


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


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
