from d2l import torch as d2l
import torch
from torch.utils import data
from torch import nn



def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


# Generate data

true_w = torch.tensor([2, -5.0])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)


print(next(iter(data_iter)))

net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr = 0.03)

num_eopchs = 3
for eopch in range(num_eopchs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {eopch + 1}, loss {l:f}')

w = net[0].weight.data
print("Estimated W = ", w)
print("True W = ", true_w)
b = net[0].bias.data
print("Estimated b = ", b)
print("True b = ", true_b)
