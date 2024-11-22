import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision
from models import Net
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

classes = [
    "Asteroid",
    "Black Hole",
    "Comet",
    "Constellation",
    "Galaxy",
    "Nebula",
    "Planet",
    "Star"
]

device = torch.device("cuda")

transform = transforms.Compose([transforms.Resize(255),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64
dataset = datasets.ImageFolder('path/to/data', transform=transform)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=16)

plt.figure(figsize = (20,10))
images, labels = next(iter(trainloader))

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
images, labels = next(iter(trainloader))

imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


net = Net().to(device)
net

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

eps = 10

for epoch in range(eps+1):
    train_loss = 0.0
    val_loss = 0.0
    net.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    net.eval()
    #TODO eval loader
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

    print(f"ep: {epoch}/{eps}, train_loss: {(train_loss/len(trainloader)):.4f}, val_loss: {(val_loss/len(trainloader)):.4f}")


print('Finished Training')


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        outputs = net(images).cpu()
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))