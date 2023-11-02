import torch
import torch.nn as nn
import torch.optim as optim
from .model import Net
from .data import testloader, testset
from .data import trainloader, trainset


def train(save_to="nn.pth", from_save=None, epochs=1, wandb=None, cuda=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() and cuda else 'cpu')
    print(device)
    net = Net()
    if from_save:
        net.load_state_dict(torch.load(from_save))
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.016, momentum=0.9)
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        correct = 0
        total = 0
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

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f} acc: {correct / total}')
        
        if wandb:
            wandb.log({"acc": correct / total, "loss": running_loss})

    print('Finished Training')
    PATH = save_to
    torch.save(net.state_dict(), PATH)