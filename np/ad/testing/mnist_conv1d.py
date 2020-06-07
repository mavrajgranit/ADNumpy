from np.ad.nn import Tensor, Conv1d, Sequential, MSE_Loss, SGD, Softmax, LeakyRelu

from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import ctypes

mkl_rt = ctypes.CDLL('mkl_rt.dll')
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads

EPOCHS = 12
BATCH_SIZE = 124*8
INPUT_SIZE = 28 * 28
OUTPUT_SIZE = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

trainset = MNIST("../data/mnist", train=True, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = MNIST("../data/mnist", train=False, transform=transform)
testloader = DataLoader(testset, batch_size=1000, shuffle=False)

model = Sequential(
    Conv1d(261, 1, channel=1, stride=(1, 1)),
    LeakyRelu(0.2),
    Conv1d(524, 1, channel=10, stride=(1, 1)),
    LeakyRelu(0.2),
    Softmax(1)
)

optim = SGD(model.parameters(), lr=0.0001)
criterion = MSE_Loss()


def train():
    l = 0
    correct = 0
    for i, (x, label) in enumerate(trainloader):
        x = Tensor(x.view(x.size(0), 1, INPUT_SIZE).numpy(), requires_grad=False)
        label = Tensor(label.numpy(), requires_grad=False).reshape((x.size(0), 1, 1))

        x_ = model(x)

        x_label = Tensor((x.size(0), OUTPUT_SIZE, 1), requires_grad=False)
        x_label.put_(label, 1, 1.0)
        preds = x_.argmax(1).reshape((x.size(0), 1, 1))
        correct += (preds == label).sum(axis=(0, 1, 2)).item()

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
        x = Tensor(x.view(x.size(0), 1, INPUT_SIZE).numpy(), requires_grad=False)
        label = Tensor(label.numpy(), requires_grad=False).reshape((x.size(0), 1, 1))

        x_ = model(x)

        x_label = Tensor((x.size(0), OUTPUT_SIZE, 1), requires_grad=False)
        x_label.put_(label, 1, 1.0)
        preds = x_.argmax(1).reshape((x.size(0), 1, 1))
        correct += (preds == label).sum(axis=(0, 1, 2)).item()

        loss = criterion(x_, x_label, axis=(0, 1, 2))
        l += loss.item()
    optim.zero_grad()
    return l / len(testset), correct / len(testset)


for e in range(EPOCHS):
    train_loss, train_acc = train()
    test_loss, test_acc = test()
    print(e, train_loss, train_acc, test_loss, test_acc)

"""np.savetxt("c1d0k.csv", model.Conv1d_0.k.value, delimiter=',')
np.savetxt("c1d0b.csv", model.Conv1d_0.b.value, delimiter=',')
np.savetxt("c1d1k.csv", model.Conv1d_1.k.value, delimiter=',')
np.savetxt("c1d1b.csv", model.Conv1d_1.b.value, delimiter=',')"""