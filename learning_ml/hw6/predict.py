import torch
from pyimagesearch.config import INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH
from pyimagesearch.model import UNet


x = torch.randn(1, 3, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)
model = UNet()
y = model(x)
print(y)