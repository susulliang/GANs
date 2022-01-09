# SRCNN Helper

import sys
import numpy as np
import torch
import torchvision.transforms as transforms
import time
from PIL import Image

sys.path.append('SRCNN')
from model import SRCNN

class upscaler:
    def __init__(self, cuda_on=False, factor=3) -> None:
        self.factor = factor
        self.last_metric = 0.0
        self.device = torch.device("cuda:0" if (
            torch.cuda.is_available() and cuda_on) else "cpu")
        self.torch_model = torch.load(
            f"SRCNN/model_{factor}x.pth").to(self.device)

    def up(self, img):
        start = time.time()
        img = Image.fromarray(img).convert('YCbCr')
        # first, we upscale the image via bicubic interpolation
        img = img.resize(
            (int(img.size[0]*self.factor), int(img.size[1]*self.factor)), Image.BICUBIC)

        y, cb, cr = img.split()

        img_to_tensor = transforms.ToTensor()
        # we only work with the "Y" channel
        input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
        input = input.to(self.device)

        out = self.torch_model(input)
        out = out.cpu()
        #out = out.cuda(non_blocking = True)

        out_img_y = out[0].detach().numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

        # we merge the output of our network with the upscaled Cb and Cr from before
        out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')
        # before converting the result in RGB
        self.last_metric = round(time.time() - start, 3)
        return out_img
