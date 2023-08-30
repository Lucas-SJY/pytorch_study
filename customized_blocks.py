import torch
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


X = torch.rand(2, 20)
net = MLP()
print("MLP", net(X))

class mySequantial(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X

net2 = mySequantial(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print("MySequential", net2(X))