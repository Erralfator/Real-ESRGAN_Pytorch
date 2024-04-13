# INSTALL

import os
import torch
import subprocess

if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")

download_path = "Real-ESRGAN_Pytorch"
if os.path.isdir(download_path):
    print(f"The directory '{download_path}' exists.")
else:
    subprocess.run("git clone https://github.com/Nick088Official/Real-ESRGAN_Pytorch.git")
os.chdir('Real-ESRGAN_Pytorch')
subprocess.run("pip install ffmpeg-python")
from RealESRGAN import RealESRGAN
from PIL import Image
import numpy as np
import ffmpeg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model2 = RealESRGAN(device, scale=2)
model2.load_weights('weights/RealESRGAN_x2.pth', download=True)
model4 = RealESRGAN(device, scale=4)
model4.load_weights('weights/RealESRGAN_x4.pth', download=True)
model8 = RealESRGAN(device, scale=8)
model8.load_weights('weights/RealESRGAN_x8.pth', download=True)
# Clearing the Screen
os.system('cls')
print(f'Installed with all its models on {"GPU" if torch.cuda.is_available() else "CPU"}!')

# RUN NO UI

import os
import shutil
from io import BytesIO
import io
from RealESRGAN import RealESRGAN
from PIL import Image
import numpy as np

print("Before choosing a model and pressing enter, be sure you putted an image inside of the 'inputs' folder inside the 'Real-ESRGAN_Pytorch' folder which you will find inside the 'Python Scripts' folder'.")

model_scale = input("Choose a model (2/4/8): ")


model = RealESRGAN(device, scale=int(model_scale))
model.load_weights(f'weights/RealESRGAN_x{model_scale}.pth', download=False)


IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

def process_input(filename):
    result_image_path = os.path.join('results/', os.path.basename(filename))
    image = Image.open(filename).convert('RGB')
    print("Processing Image/s Input...")
    sr_image = model.predict(np.array(image))
    sr_image.save(result_image_path)
    print(f'Finished! Image saved to {result_image_path}')

# Process all files in the 'inputs' folder
input_folder = 'inputs'
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        input_file_path = os.path.join(input_folder, filename)
        process_input(input_file_path)
