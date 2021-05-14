import torch
import torchvision
from torchvision import transforms
from UNet import UNet
from dataset import CustomImageDataset, NoisyDataset
from utils import imshow
import argparse

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--var', type=float, default=.5)
parser.add_argument('--data_dir', type=str, default='./data/test')
parser.add_argument('--checkpoint', type=str,
                    default='./checkpoints/chckpt_gamma0_var_35.pt')

transform = transforms.Compose(
    [transforms.CenterCrop(256),
     transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

args = parser.parse_args()

VAR = args.var
DATA_DIR = args.data_dir
CHECKPOINT = args.checkpoint

testset = CustomImageDataset(DATA_DIR, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4)
dataiter = iter(testloader)
checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu'))


model_test = UNet(in_channels=3, out_channels=3).double()
model_test.load_state_dict(checkpoint['model_state_dict'])
model_test = model_test.cpu()
model_test.train()

noisy = NoisyDataset(var=VAR)

images, _ = dataiter.next()
noisy_images = noisy(images)
# Displaying the Noisy Images
imshow(torchvision.utils.make_grid(noisy_images.cpu()))
# Displaying the Denoised Images
imshow(torchvision.utils.make_grid(model_test(noisy_images.cpu())))
