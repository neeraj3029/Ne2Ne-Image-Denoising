import requests
import cv2
from PIL import Image

file = '/Users/neeraj/denoiser/data/test/4.png'
 
resp = requests.post("http://localhost:3000/predict",
                     files={"file": open(file,'rb')})

with open("response.jpg", "wb") as f:
    f.write(resp.content)
