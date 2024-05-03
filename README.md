# Real-ESRGAN Pytorch

[![Discord](https://img.shields.io/discord/1198701940511617164?color=%23738ADB&label=Discord&style=for-the-badge)](https://discord.gg/osai)

PyTorch implementation of a Real-ESRGAN model trained on custom dataset. This model shows better results on faces compared to the original version. It is also easier to integrate this model into your projects.

This is a forked Github Version which also includes inferencing for videos

> This is not an official implementation. We partially use code from the [original repository](https://github.com/xinntao/Real-ESRGAN)

Real-ESRGAN is an upgraded [ESRGAN](https://arxiv.org/abs/1809.00219) trained with pure synthetic data is capable of enhancing details while removing annoying artifacts for common real-world images. 



- [Paper (Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data)](https://arxiv.org/abs/2107.10833)
- [Original implementation](https://github.com/xinntao/Real-ESRGAN)
- [Pytorch Original implementation](https://github.com/ai-forever/Real-ESRGAN)
- [Huggingface Page Models 🤗](https://huggingface.co/ai-forever/Real-ESRGAN)

## Examples

---

Low quality image:

![](inputs/lr_image.png)

Real-ESRGAN result:

![](results/sr_image.png)

---

Low quality image:

![](inputs/lr_face.png)

Real-ESRGAN result:

![](results/sr_face.png)

---

Low quality image:

![](inputs/lr_lion.png)

Real-ESRGAN result:

![](results/sr_lion.png)

## Usage

---

### Basic Python Local Usage:

##### Installation

```bash
pip install git+https://github.com/Nick088/Real-ESRGAN_Pytorch.git
```

##### Inference

```python
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

path_to_image = 'inputs/lr_image.png'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('results/sr_image.png')
```

### Precompiled:

1. Click Code and Download as Zip.
2. Open Run_Inference_UI.bat.
3. After it installs the dependencies, let it run and do Ctrl+Click on Local URL.
4. Now either for Image Tab or Video Tab, just upload your input, select the Upscale Model (x2, x4, x8) and Submit!

---

### Online Usage:

#### Google Colab:
- No UI <a target="_blank" href="https://colab.research.google.com/github/Nick088Official/Real-ESRGAN_Pytorch/blob/main/Real_ESRGAN_Pytorch_Inference_NO_UI.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- WEB UI **(WARNING: COULD RISK YOUR COLAB FREE TIER)** <a target="_blank" href="https://colab.research.google.com/github/Nick088Official/Real-ESRGAN_Pytorch/blob/main/Real_ESRGAN_Pytorch_Inference_WEB_UI.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### Hugging Face Space:
- [Huggingface Space🤗](https://huggingface.co/spaces/Nick088/Real-ESRGAN_Pytorch)

#### Kaggle:
- [WEB UI **(WARNING: YOU COULD RISK YOUR KAGGLE ACCOUNT)** ![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/nick088/real-esrgan-pytorch-inference-web-ui)

### CHANGELOG
#### Update - May 3rd, 2024
Added web ui precompiled local, kaggle web ui notebook

### Credits:
- [Nick088](https://linktr.ee/Nick088) (improving and porting)
- [forever-ai](https://huggingface.co/ai-forever/Real-ESRGAN) (making the pytorch implementation model of Real-ESRGAN)
- Xinntao
- Tencent
- Geeve George (no ui inference video code)
- daroche (fixing web ui video inference code)
