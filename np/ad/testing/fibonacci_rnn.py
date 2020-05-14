from np.ad.nn import Tensor, Variable, Linear, MSE_Loss, Sigmoid, SGD, Sequential, Module, Tanh


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
        return Tensor((1, self.l.outputs), requires_grad=False)


class RNN(Module):

    def __init__(self, w, b, m, s=0):
        super(RNN, self).__init__()
        self.w = Tensor(w)
        self.b = Tensor(b)
        self.m = Tensor(m)
        self.s = s

    def forward(self, x, hidden):
        o = self.w @ x + self.b
        h = self.m @ hidden
        y = o + h
        return y, y

    def init_hidden(self):
        return Tensor([self.s], requires_grad=False)

rnn = RNN([1.0], [0.0], [1.0])
lr = 0.000000001
optim = SGD(rnn.parameters(), lr=lr)
inputs = [

]

targets = [

]

def generate_fib(n, c=1.0, prev=0, cc=0):
    if cc == 0:
        inputs.append(Tensor([1.0], requires_grad=False))
        f = 1.0
        c = 0.0
    else:
        f = c + prev
        inputs.append(Tensor([f], requires_grad=False))

    cc += 1
    if cc <= n:
        generate_fib(n, f, c, cc)


train = 21
test = 3

generate_fib(train + test)
targets = inputs[1:len(inputs)]
del inputs[len(inputs) - 1]

print(targets)

for e in range(500):
    l = 0
    hidden = rnn.init_hidden()
    for i in range(len(inputs) - test):
        y, hidden = rnn(inputs[i], hidden)
        loss = ((targets[i] - y) ** 2 / 2).sum()
        loss.backward()
        l += loss
    optim.step()
    optim.zero_grad()
    print(e, l)

hidden = rnn.init_hidden()
l = 0
for i in range(len(inputs)):
    y, hidden = rnn(inputs[i], hidden)
    l += ((targets[i] - y) ** 2 / 2).sum()
    print(y)
print(l)
print(rnn.w, rnn.b, rnn.m)