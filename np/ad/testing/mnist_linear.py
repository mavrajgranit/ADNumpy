from np.ad.nn import Linear, MSE_Loss, Sequential, Sigmoid, SGD, Tensor, Relu, LeakyRelu, Tanh, Softmax
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import numpy as np
import ctypes

mkl_rt = ctypes.CDLL('mkl_rt.dll')
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads

EPOCHS = 17
BATCH_SIZE = 4
INPUT_SIZE = 28*28
OUTPUT_SIZE = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

trainset = MNIST("../data/mnist", train=True, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = MNIST("../data/mnist", train=False, transform=transform)
testloader = DataLoader(testset, batch_size=len(testset), shuffle=False)

model = Sequential([
    Linear(INPUT_SIZE, 256),
    LeakyRelu(0.2),
    Linear(256, 128),
    LeakyRelu(0.2),
    Linear(128, OUTPUT_SIZE),
    Softmax()
])

optim = SGD(model, lr=0.01)
criterion = MSE_Loss()


def train():
    l = 0
    correct = 0
    for i, (x, label) in enumerate(trainloader):
        x = Tensor(x.view(-1, INPUT_SIZE).numpy())
        label = Tensor(label.numpy()).reshape((label.size(0), 1))

        x_ = model(x)
        x_label = Tensor((x_.shape[0], OUTPUT_SIZE))
        x_label.put_(label, 1, 1.0)
        preds = x_.argmax(1).reshape((label.shape[0], 1))
        correct += (preds == label).sum().item()

        loss = criterion(x_, x_label, axis=(0, 1))
        l += loss

        loss.backward()

        optim.step()
        optim.zero_grad()
    return l / len(trainset), correct / len(trainset)


def test():
    l = 0
    correct = 0
    for i, (x, label) in enumerate(testloader):
        x = Tensor(x.view(-1, INPUT_SIZE).numpy())
        label = label.numpy().reshape(label.size(0), 1)

        x_ = model(x)
        x_label = Tensor((x_.shape[0], OUTPUT_SIZE))
        x_label.put_(label, 1, 1.0)
        preds = x_.argmax(1).reshape((label.shape[0], 1))
        correct += (preds == label).sum().item()

        loss = criterion(x_, x_label, axis=(0, 1))
        l += loss
    return l / len(testset), correct / len(testset)


for e in range(EPOCHS):
    train_loss, train_acc = train()
    test_loss, test_acc = test()
    print(e, train_loss, train_acc, test_loss, test_acc)

np.savetxt("l0w.csv", model.Linear_0.w.value, delimiter=',')
np.savetxt("l0b.csv", model.Linear_0.b.value, delimiter=',')
np.savetxt("l1w.csv", model.Linear_1.w.value, delimiter=',')
np.savetxt("l1b.csv", model.Linear_1.b.value, delimiter=',')
np.savetxt("l2w.csv", model.Linear_2.w.value, delimiter=',')
np.savetxt("l2b.csv", model.Linear_2.b.value, delimiter=',')