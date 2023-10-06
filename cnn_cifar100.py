import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary


# 定义模型
class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=2), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=2), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=2), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
            #nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            #nn.Linear(2048, )
            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(0.55),
            nn.Linear(4096, 5120), nn.ReLU(), nn.Dropout(0.55),
            nn.Linear(5120, 4096), nn.Sigmoid(), nn.Dropout(0.55),
            nn.Linear(4096, 2048), nn.Dropout(0.5),
            nn.Linear(2048, 100)
        )

    def forward(self, x):
        x = self.feature(x)
        output = self.classifier(x)
        return output



net = CNNCifar()
#print(net)

# 加载数据集
apply_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
])

train_dataset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True,
                                             transform=apply_transform)
test_dataset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=False,
                                            transform=apply_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0015,weight_decay=5e-8, amsgrad=True)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 如果有gpu就使用gpu，否则使用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

# 训练模型
print('training on: ', device)

def test():
    net.eval()
    acc = 0.0
    sum = 0.0
    loss_sum = 0
    for batch, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = net(data)
        loss = criterion(output, target)
        acc += torch.sum(torch.argmax(output, dim=1) == target).item()
        sum += len(target)
        loss_sum += loss.item()
    print('test  acc: %.2f%%, loss: %.4f' % (100 * acc / sum, loss_sum / (batch + 1)))


def train():
    net.train()
    acc = 0.0
    sum = 0.0
    loss_sum = 0
    for batch, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        acc += torch.sum(torch.argmax(output, dim=1) == target).item()
        sum += len(target)
        loss_sum += loss.item()

        if batch % 200 == 0:
            print('\tbatch: %d, loss: %.4f' % (batch, loss.item()))
    print('train acc: %.2f%%, loss: %.4f' % (100 * acc / sum, loss_sum / (batch + 1)))


'''for batch, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    print(data.size())'''
print(summary(net, ( 3, 32, 32)))
for epoch in range(60):
    print('epoch: %d' % epoch)
    train()
    test()
    #,
            #nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2),