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
    def __init__(self, cuda_on = False) -> None:
        self.last_metric = 0.0
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and cuda_on) else "cpu")
        self.torch_model = torch.load("SRCNN/model_2x.pth").to(self.device)


    def res2x(self, img):
        start = time.time()
        img = Image.fromarray(img).convert('YCbCr')
        img = img.resize((int(img.size[0]*2), int(img.size[1]*2)), Image.BICUBIC)  # first, we upscale the image via bicubic interpolation
        y, cb, cr = img.split()

        img_to_tensor = transforms.ToTensor()
        input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])  # we only work with the "Y" channel

        input = input.to(self.device)

        out = self.torch_model(input)
        out = out.cpu()
        out_img_y = out[0].detach().numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
        
        out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')  # we merge the output of our network with the upscaled Cb and Cr from before
                                                           # before converting the result in RGB
        self.last_metric = round(time.time() - start, 3)
        return out_img


