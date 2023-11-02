import torch
import torch.nn as nn
import torch.optim as optim
from model import Net
from data import testloader, testset
from data import trainloader, trainset

import sys
save_to = "nn.path"
for i in range(len(sys.argv)):
    if sys.argv[i] == "--save":
        if i < len(sys.argv) - 1:
            save_to = sys.argv[i + 1]

from_save = None
for i in range(len(sys.argv)):
    if sys.argv[i] == "--from":
        if i < len(sys.argv) - 1:
            from_save = sys.argv[i + 1]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = Net()
if from_save:
    net.load_state_dict(torch.load(from_save))
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
PATH = save_to
torch.save(net.state_dict(), PATH)