from utils import config, ToTensor
import glob, os
import torch
import matplotlib.pyplot as plt
from utils import dataloader
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from src.model_AE import *
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = config()


dataload = dataloader.Dataloader(config,transform=transforms.Compose([ToTensor()]))
dataloadered = DataLoader(dataload, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)

num_epochs = config.train.num_epochs
learning_rate = config.train.learning_rate
net = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
net.parameters(), lr=learning_rate, weight_decay=config.train.weight_decay)



for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloadered, 1):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data["input"].float(), data["label"].float()

        if i==1:
            datasample = inputs
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

#sample = dataload[1]
#datasample = sample['input']
print(datasample.shape)
x_latent = net.encoder(datasample)
print(x_latent.shape)

def plot_slidder(x_latent, net):
    pass
x_latent_modify = x_latent + 1
sortie_to_plot = net.decoder(x_latent_modify)

"""
fig = plt.figure()

for i in range(len(dataload)):
    sample = dataload[i+1]
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample["input"])

    if i == 3:
        plt.show()
        break


"""
