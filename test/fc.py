#!/usr/bin/python3
import numpy as np
from minl import Adam, Tensor

import torch
from torchvision import datasets, transforms

class FcNet:
    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.w = Tensor(np.random.rand(dim_out, dim_in) - 0.5)
        self.b = Tensor(np.random.rand(dim_out, dim_in) - 0.5)
        self.params = [self.w, self.b]

    def forward(self, data):
        x = Tensor(data.flatten())
        x = x.broadcast(self.w.shape)
        x = x * self.w
        x = x + self.b
        x = x.relu()
        x = x.sum(axis=1)
        x = x.relu()
        return x

# target is dimension in one-hot encoding.
def simple_loss(output, target):
    one_hot = np.zeros(output.shape)
    one_hot.flat[target] = 1.
    # print("one_hot: {}".format(one_hot))
    target_v = Tensor(one_hot)

    total = output.sum()
    # print("total: {}".format(total.value))
    total = total.broadcast(output.shape)
    # print("total: {}".format(total.value))
    prob = output * total.inv()
    # print("prob: {}".format(prob.value))

    diff = target_v - prob
    # print("diff: {}".format(diff.value))
    loss = (diff * diff).sum()
    # print("loss: {}".format(loss.value))

    return loss

def train(model, train_loader, optimizer, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model.forward(data)
        loss = simple_loss(output, target)
        loss.backward()
        # print("w grad: {}".format(model.w.grad))
        # print("b grad: {}".format(model.b.grad))
        optimizer.step()
        if batch_idx % 1000 == 0:
            print(model.w.value)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.value[0,0]))

def test(model, test_loader):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model.forward(data)
        loss = simple_loss(output, target).value[0,0]
        test_loss += loss
        pred = output.value.argmax(axis=0)
        if pred[0] == target[0]:
            correct += 1

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def get_datasets():
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1)
    test_loader = torch.utils.data.DataLoader(dataset2)
    return (train_loader, test_loader)

def main():
    (train_loader, test_loader) = get_datasets()

    net = FcNet(28*28, 10)

    optimizer = Adam(net.params)
    for epoch in range(1):
        train(net, train_loader, optimizer, epoch)
        test(net, test_loader)

if __name__ == "__main__":
    main()
