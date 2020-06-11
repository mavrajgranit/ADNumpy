from np.ad.nn import Tensor, MSE_Loss
import np.ad.functional as F
from np.ad.operation import conv2d
import numpy as np

cx = 4
cy = 4
cz = 2
batch = 2
channel = 2
kx, ky, kz = 2, 2, 1
sx, sy, sz = 1, 1, 1
input = Tensor(np.arange(1, batch * cz * cy * cx + 1, dtype=np.float).reshape(batch, cz, cy, cx))
k = Tensor(np.arange(1, channel * kz * ky * kx + 1, dtype=np.float).reshape(channel, kz * ky * kx))
tar_k = Tensor(k.value*2.0)
v, _, _ = F.conv2d(input.value, tar_k.value, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))
tar_y = Tensor(v)

mse = MSE_Loss()
for e in range(0):
    y = conv2d.forward(input, k,  kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))
    loss = mse(y, tar_y, (0, 1, 2, 3))
    loss.backward()
    k.value -= k.gradient * 0.000001
    #print(k.gradient)
    k.zero_grad()

    #input.value -= input.gradient * 0.001
    #print(input.gradient)
    #input.zero_grad()

    print(e, loss.item())
print(tar_y-y)
print(input)
print(k)

from np.ad.operation import conv2d_transpose

cx = 2
cy = 2
cz = 2
batch = 2
channel = 2
kx, ky, kz = 2, 2, 1
sx, sy, sz = 1, 1, 1
input = Tensor(np.arange(1, batch * cz * cy * cx + 1, dtype=np.float).reshape(batch, cz, cy, cx))
k = Tensor(np.arange(1, channel * kz * ky * kx + 1, dtype=np.float).reshape(channel, kz * ky * kx))
tar_k = Tensor(k.value*2.0)
v, _, _ = F.conv2d_transpose(input.value, tar_k.value, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))
tar_y = Tensor(v)

mse = MSE_Loss()
for e in range(25000):
    y = conv2d_transpose.forward(input, k,  kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))
    loss = mse(y, tar_y, (0, 1, 2, 3))
    loss.backward()
    k.value -= k.gradient * 0.000001
    #print(k.gradient)
    k.zero_grad()

    #input.value -= input.gradient * 0.001
    #print(input.gradient)
    #input.zero_grad()

    print(e, loss.item())
print(tar_y-y)
print(input)
print(k)