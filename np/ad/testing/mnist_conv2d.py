from np.ad.nn import Tensor, Conv2d, ReshapeBatch, Linear, Sequential, MSE_Loss, SGD, Softmax, LeakyRelu, Sigmoid

from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import ctypes

mkl_rt = ctypes.CDLL('mkl_rt.dll')
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads

EPOCHS = 10
BATCH_SIZE = 124*5
INPUT_SIZE = 28 * 28
OUTPUT_SIZE = 10

transform = transforms.Compose([
    #transforms.RandomAffine(30, translate=(0.214, 0.214)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

trainset = MNIST("../data/mnist", train=True, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = MNIST("../data/mnist", train=False, transform=transform)
testloader = DataLoader(testset, batch_size=600, shuffle=False)

model = Sequential(
    Conv2d(2, 2, 1, channel=2, stride=(1, 1, 1)),
    LeakyRelu(0.2),
    Conv2d(3, 3, 2, channel=4, stride=(2, 2, 1)),
    LeakyRelu(0.2),
    Conv2d(4, 4, 4, channel=8, stride=(1, 1, 1)),
    LeakyRelu(0.2),
    ReshapeBatch(10*10*8),
    Linear(10*10*8, 1024),
    LeakyRelu(0.2),
    Linear(1024, 128),
    LeakyRelu(0.2),
    Linear(128, 10),
    Softmax(2)
)

optim = SGD(model.parameters(), lr=1e-3)
criterion = MSE_Loss()


def train():
    l = 0
    correct = 0
    for i, (x, label) in enumerate(trainloader):
        x = Tensor(x.numpy(), requires_grad=False)
        label = Tensor(label.numpy(), requires_grad=False).reshape((x.size(0), 1, 1))

        x_ = model(x)

        x_label = Tensor((x.size(0), 1, OUTPUT_SIZE), requires_grad=False)
        x_label.put_(label, 2, 1.0)
        preds = x_.argmax(2).reshape((x.size(0), 1, 1))
        correct += (preds == label).sum(axis=(0, 1, 2)).item()
        #print(correct)

        loss = criterion(x_, x_label, axis=(0, 1, 2))
        l += loss.item()

        loss.backward()
        optim.step()
        optim.zero_grad()
    return l / len(trainset), correct / len(trainset)


def test():
    l = 0
    correct = 0
    for i, (x, label) in enumerate(testloader):
        x = Tensor(x.numpy(), requires_grad=False)
        label = Tensor(label.numpy(), requires_grad=False).reshape((x.size(0), 1, 1))

        x_ = model(x)

        x_label = Tensor((x.size(0), 1, OUTPUT_SIZE), requires_grad=False)
        x_label.put_(label, 2, 1.0)
        preds = x_.argmax(2).reshape((x.size(0), 1, 1))
        correct += (preds == label).sum(axis=(0, 1, 2)).item()

        loss = criterion(x_, x_label, axis=(0, 1, 2))
        l += loss.item()
    optim.zero_grad()
    return l / len(testset), correct / len(testset)


for e in range(EPOCHS):
    train_loss, train_acc = train()
    test_loss, test_acc = test() if e > 2 else 0, 0
    print(e, train_loss, train_acc, test_loss, test_acc)

import numpy as np

np.save("conv2d/c2d0k", model.Conv2d_0.k.value)
np.save("conv2d/c2d0b", model.Conv2d_0.b.value)
np.save("conv2d/c2d1k", model.Conv2d_1.k.value)
np.save("conv2d/c2d1b", model.Conv2d_1.b.value)
np.save("conv2d/c2d2k", model.Conv2d_2.k.value)
np.save("conv2d/c2d2b", model.Conv2d_2.b.value)
np.save("conv2d/lin0w", model.Linear_0.w.value)
np.save("conv2d/lin0b", model.Linear_0.b.value)
np.save("conv2d/lin1w", model.Linear_1.w.value)
np.save("conv2d/lin1b", model.Linear_1.b.value)
np.save("conv2d/lin2w", model.Linear_2.w.value)
np.save("conv2d/lin2b", model.Linear_2.b.value)