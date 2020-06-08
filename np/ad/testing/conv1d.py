from np.ad.nn import Tensor, MSE_Loss
from np.ad.operation import conv1d
import numpy as np

inp = []
inp.append([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]])
inp.append([[2.0, 4.0, 6.0],
            [8.0, 10.0, 12.0],
            [14.0, 16.0, 18.0]])

input = Tensor(np.array(inp), requires_grad=True)
k = Tensor(np.array([[1.0, 2.0, 1.0, 5.0],
                     [1.0, 3.0, 4.0, 1.0]]))

tar = []
tar.append([[12.0, 16.0],
           [24.0, 28.0],
           [24.0, 32.0],
           [48.0, 56.0]])
tar.append([[24.0, 32.0],
           [48.0, 56.0],
           [48.0, 64.0],
           [56.0, 112.0]])
target = Tensor(np.array(tar))

mse = MSE_Loss()
for e in range(30000):
    y = conv1d.forward(input, k, kx=2, ky=2, channel=2, stride=(1, 1))
    loss = mse(y, target, (0, 1, 2))
    loss.backward()
    k.value -= k.gradient * 0.0001
    #print(k.gradient)
    k.zero_grad()

    input.value -= input.gradient * 0.001
    #print(input.gradient)
    input.zero_grad()

    print(loss.item())
print(y)
print(input)
print(k)

from np.ad.operation import conv1d_transpose

inp = []

inp.append([[1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]])
inp.append([[2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0]])

input = Tensor(np.array(inp), requires_grad=True)
k = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)

tar = []
tar.append([[2.0, 5.0, 8.0, 3.0],
            [8.0, 14.0, 17.0, 6.0]])
tar.append([[4.0, 10.0, 16.0, 6.0],
            [16.0, 28.0, 34.0, 12.0]])
target = Tensor(np.array(tar))

mse = MSE_Loss()
for e in range(30000):
    y = conv1d_transpose.forward(input, k, kx=2, ky=1, channel=1, stride=(1, 1))

    loss = mse(y, target, (0, 1, 2))
    loss.backward()
    k.value -= k.gradient * 0.0001
    #print(k.gradient)
    k.zero_grad()

    input.value -= input.gradient * 0.001
    #print(input.gradient)
    input.zero_grad()

    print(loss.item())
print(y)
print(input)
print(k)