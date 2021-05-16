import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from dataset import CustomImageDataset, NoisyDataset
from UNet import UNet
from utils import imshow


class Neighbour2Neighbour():
    def __init__(self, gamma=2, k=2):
        self.gamma = gamma
        self.k = k
        self.EPOCHS, self.BATCH, self.VAR, self.LR, self.DATA_DIR, self.CH_DIR = self.__get_args__()
        self.transforms = transforms.Compose(
            [transforms.CenterCrop(256),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])
        self.trainloader, self.validloader = self.load_data()
        self.use_cuda = torch.cuda.is_available()

    def __get_args__(self):
        parser = argparse.ArgumentParser(description='Parameters')
        parser.add_argument('--epochs', type=int, default=15)
        parser.add_argument('--batch', type=int, default=4)
        parser.add_argument('--var', type=float, default=.5)
        parser.add_argument('--learning_rate', type=float, default=.0005)
        parser.add_argument('--data_dir', type=str, default='./data')
        parser.add_argument('--checkpoint_dir', type=str,
                            default='./checkpoints')

        args = parser.parse_args()
        return (args.epochs, args.batch, args.var, args.learning_rate, args.data_dir, args.checkpoint_dir)

    def subsample(self, image):
        # This function only works for k = 2 as of now.
        blen, channels, m, n = np.shape(image)
        dim1, dim2 = m // self.k, n // self.k
        image1, image2 = np.zeros([blen, channels, dim1, dim2]), np.zeros(
            [blen, channels, dim1, dim2])

        image_cpu = image.cpu()
        for channel in range(channels):
            for i in range(dim1):
                for j in range(dim2):
                    i1 = i * self.k
                    j1 = j * self.k
                    num = np.random.choice([0, 1, 2, 3])
                    if num == 0:
                        image1[:, channel, i, j], image2[:, channel, i, j] = image_cpu[:,
                                                                                       channel, i1, j1], image_cpu[:, channel, i1, j1+1]
                    elif num == 1:
                        image1[:, channel, i, j], image2[:, channel, i, j] = image_cpu[:,
                                                                                       channel, i1+1, j1], image_cpu[:, channel, i1+1, j1+1]
                    elif num == 2:
                        image1[:, channel, i, j], image2[:, channel, i, j] = image_cpu[:,
                                                                                       channel, i1, j1], image_cpu[:, channel, i1+1, j1]
                    else:
                        image1[:, channel, i, j], image2[:, channel, i, j] = image_cpu[:,
                                                                                       channel, i1, j1+1], image_cpu[:, channel, i1+1, j1+1]

        if self.use_cuda:
            return torch.from_numpy(image1).cuda(), torch.from_numpy(image2).cuda()
        return torch.from_numpy(image1), torch.from_numpy(image2)

    def load_data(self):
        trainset = CustomImageDataset(
            self.DATA_DIR + '/train/', transform=self.transforms)
        validset = CustomImageDataset(
            self.DATA_DIR + '/val/', transform=self.transforms)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.BATCH, num_workers=2, shuffle=True,)
        validloader = torch.utils.data.DataLoader(
            validset, batch_size=self.BATCH, num_workers=2, shuffle=True,)
        return trainloader, validloader

    def get_model(self):
        model = UNet(in_channels=3, out_channels=3).double()
        if self.use_cuda:
            model = model.cuda()
        noisy = NoisyDataset(var=self.VAR)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)
        criterion = RegularizedLoss()
        return model, noisy, optimizer, criterion

    def train(self):
        model, noisy, optimizer, criterion = self.get_model()
        if self.use_cuda:
            model = model.cuda()

        min_loss_valid = 100
        for epoch in range(self.EPOCHS):
            total_loss_valid = 0
            total_loss = 0
            for idx, (batch, _) in enumerate(self.trainloader):
                optimizer.zero_grad()
                noisy_image = noisy(batch)
                if self.use_cuda:
                    noisy_image = noisy_image.cuda()
                g1, g2 = self.subsample(noisy_image)
                fg1 = model(g1)
                with torch.no_grad():
                    X = model(noisy_image)
                    G1, G2 = self.subsample(X)
                total_loss = criterion(fg1, g2, G1, G2)
                total_loss.backward()
                optimizer.step()

            for idx, (batch, _) in enumerate(self.validloader):
                with torch.no_grad():
                    noisy_image = noisy(batch)
                    if self.use_cuda:
                        noisy_image = noisy_image.cuda()
                    g1, g2 = self.subsample(noisy_image)
                    fg1 = model(g1)
                    X = model(noisy_image)
                    G1, G2 = self.subsample(X)
                    total_loss_valid = criterion(fg1, g2, G1, G2)

            if total_loss_valid < min_loss_valid:
                min_loss_valid = total_loss_valid

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, self.CH_DIR + '/chk_' + str(self.k) + '_' + str(self.gamma)+'_'+str(self.VAR)+'.pt')
                print('Saving Model...')
            print('Epoch', epoch+1, 'Loss Valid:',
                  total_loss_valid, 'Train', total_loss)


class RegularizedLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()

        self.gamma = gamma

    def mseloss(self, image, target):
        x = ((image - target)**2)
        return torch.mean(x)

    def regloss(self, g1, g2, G1, G2):
        return torch.mean((g1-g2-G1+G2)**2)

    def forward(self, fg1, g2, G1f, G2f):
        return self.mseloss(fg1, g2) + self.gamma * self.regloss(fg1, g2, G1f, G2f)


if __name__ == '__main__':
    N2N = Neighbour2Neighbour(gamma=1)
    N2N.train()
