import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import numpy

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
if __name__ == '__main__':
    trainset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform = transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
          self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
          self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
          self.pool = nn.MaxPool2d(2, 2)
          self.fc1 = nn.Linear(128 * 4 * 4, 512)
          self.fc2 = nn.Linear(512, 128)
          self.fc3 = nn.Linear(128, 10)
          self.dropout = nn.Dropout(0.5)

      def forward(self, x):
          x = self.pool(F.relu(self.conv1(x)))
          x = self.pool(F.relu(self.conv2(x)))
          x = self.pool(F.relu(self.conv3(x)))
          x = x.view(-1, 128 * 4 * 4)
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          return x
            
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr = 0.001, momentum=0.9)
    net.train();
    for epoch in range(1):
        running_loss=0.0
        for i,data in enumerate(trainloader, 0):
            inputs,labels=data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if i%2000 == 1999:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Training complete')

    correct = 0
    total = 0
    net.eval();
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total+=labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
