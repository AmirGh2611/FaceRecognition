from torch import nn
from model import ConvNet
from torch import optim
from data_prep import train_loader, valid_loader
import torch
import matplotlib.pyplot as plt

Net = ConvNet()
loss_fn = nn.BCELoss()
optimizer = optim.SGD(Net.parameters(), lr=0.2)
epoch = 50
loss_value = []
Net.train()
for epoch in range(epoch):
    for images, labels in train_loader:
        optimizer.zero_grad()
        img1 = images[:, 0, :, :, :]
        img2 = images[:, 1, :, :, :]
        output1 = Net(img1)
        output2 = Net(img2)
        outputs = torch.norm(output1 - output2, p=2, dim=1, keepdim=True)  # Euclidean distance
        outputs = Net.layer6(outputs)
        loss = loss_fn(outputs, labels)
        loss_value.append(loss.item())
        print(f"epoch:{epoch}, loss:{loss.item()} ")
        loss.backward()
        optimizer.step()
plt.plot(loss_value)
plt.show()
