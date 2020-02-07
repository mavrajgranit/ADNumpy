from np.ad.nn import Tensor, Linear, MSE_Loss, Sigmoid, SGD, Sequential, Module, Tanh


class RNN(Module):

    def __init__(self, inputs, outputs):
        super().__init__()
        self.l = Linear(inputs, outputs)
        self.m = Linear(outputs, outputs, bias=False)

    def forward(self, x, hidden):
        l = self.l(x)
        m = self.m(hidden)
        y = l+m
        return y, l

    def init_hidden(self):
        return Tensor((1, self.l.outputs), rand_init=False)

inputs = [Tensor([[1.0]]), Tensor([[1.0]]), Tensor([[2.0]]), Tensor([[3.0]]), Tensor([[5.0]]), Tensor([[8.0]]), Tensor([[13.0]]), Tensor([[21.0]]), Tensor([[34.0]]), Tensor([[55.0]])]
targets = [Tensor([[1.0]]), Tensor([[2.0]]), Tensor([[3.0]]), Tensor([[5.0]]), Tensor([[8.0]]), Tensor([[13.0]]), Tensor([[21.0]])]

rnn = RNN(1, 1)
lr = 0.001
optimizer = SGD(rnn, lr=lr)
criterion = MSE_Loss()

for e in range(20000):

    l = 0
    hidden = rnn.init_hidden()
    for i in range(len(inputs)-5):
        x = inputs[i]
        t = targets[i]
        y, hidden = rnn(x, hidden)

        loss = criterion(y, t, keepdims=True)
        loss.backward()
        l += loss.item()

    optimizer.step()
    optimizer.zero_grad()
    print(e, l)

hidden = rnn.init_hidden()
for i in range(len(inputs)):
    x = inputs[i]
    y, hidden = rnn(x, hidden)
    print(y)

print(rnn.l.w)
print(rnn.l.b)
print(rnn.m.w)