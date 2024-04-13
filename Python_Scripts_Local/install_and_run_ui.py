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
    subprocess.run("git clone https://huggingface.co/spaces/Nick088/Real-ESRGAN_Pytorch")
os.chdir('Real-ESRGAN_Pytorch')
subprocess.run("pip install -r requirements.txt")

# Clearing the Screen
os.system('cls')
print(f'Installed on {"GPU" if torch.cuda.is_available() else "CPU"}!')

# RUN UI

subprocess.run(["python app.py"])