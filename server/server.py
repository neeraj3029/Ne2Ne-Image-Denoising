import io
import json
import cv2
import numpy as np

import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from UNet import UNet

import torch
import torchvision
from torchvision import transforms
from UNet import UNet
from dataset import CustomImageDataset, NoisyDataset
from utils import imshow, convert_to_numpy

app = Flask(__name__)

noisy = NoisyDataset(var=0.00000001)
CHECKPOINT = './checkpoints/chckpt_lr_0005_30_epchs_1gamma0_var_35.pt'
checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu'))
model_test = UNet(in_channels=3, out_channels=3).double()
model_test.load_state_dict(checkpoint['model_state_dict'])
# model_test.double()
model_test.eval()


transform = transforms.Compose(
    [transforms.CenterCrop(256),
     transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

def transform_image(file):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5), (0.5))
                                        ])

    image = Image.open(io.BytesIO(file)).convert("RGB")

    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    image_bytes = noisy(image_bytes.type(torch.FloatTensor)).cpu()
    outputs = model_test.forward(image_bytes)
    return image_bytes

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        tensor = transform_image(file=img_bytes)
        foo = get_prediction(image_bytes=tensor)
        npimg = convert_to_numpy(foo[0])
        img_float32 = np.float32(npimg)
        im_rgb = cv2.cvtColor(img_float32, cv2.COLOR_BGR2RGB)
        success, encoded_image = cv2.imencode('.png', im_rgb)
        content2 = encoded_image.tobytes()

        return content2


if __name__ == '__main__':
    DATA_DIR = './data/test'
    app.run(port=3000)
