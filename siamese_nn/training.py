from torch import nn
from model import ConvNet
from torch import optim
from data_prep import train_loader, valid_loader
import torch
import matplotlib.pyplot as plt
Net = ConvNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(Net.parameters(), lr=0.2)
epoch = 50
loss_value = []
Net.train()
for epoch in range(epoch):
    for images, labels in train_loader:
        optimizer.zero_grad()
        output1 = Net(images[:, 0, :, :, :])
        output2 = Net(images[:, 1, :, :, :])
        outputs = torch.cdist(output1, output2, p=2)
        loss = loss_fn(outputs, labels)
        loss_value.append(loss.item())
        print(f"epoch:{} ")
        loss.backward()
plt.plot(loss_value)
plt.show()