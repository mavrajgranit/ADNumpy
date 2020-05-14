from np.ad.nn import Tensor, Linear, MSE_Loss, Sigmoid, SGD, Sequential, Variable

l1 = Linear(2, 2)
l1.w = Variable(Tensor([[1.0, 1.0], [2.0, 2.0]]))
l1.b = Variable(Tensor([[1.0, 1.0]]))
l2 = Linear(2, 1)
l2.w = Variable(Tensor([[1.0], [2.0]]))
l2.b = Variable(Tensor([[1.0]]))
sigmoid = Sigmoid()
model = Sequential(l1, sigmoid, l2)

lr = 0.125
optimizer = SGD(model.parameters(), lr=lr)
criterion = MSE_Loss()

# inputs = [Tensor([[0.0, 0.0]]), Tensor([[0.0, 1.0]]), Tensor([[1.0, 0.0]]), Tensor([[1.0, 1.0]])]
# targets = [Tensor([[0.0]]), Tensor([[1.0]]), Tensor([[1.0]]), Tensor([[0.0]])]
inputs = [Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], requires_grad=False)]
targets = [Tensor([[0.0], [1.0], [1.0], [0.0]], requires_grad=False)]

for e in range(1800):
    l = 0
    for i in range(len(inputs)):
        x = model(inputs[i])

        loss = criterion(x, targets[i])
        l += loss.value.item()
        loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    print(e, l)

for i in range(len(inputs)):
    x = model(inputs[i])
    print(x)
