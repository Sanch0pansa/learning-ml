from hw4.train import train
from hw4.test import test

train(to_load="nn.pth")
test(from_save="nn.pth")