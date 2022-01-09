# Base VQGAN CLIP functions and helpers
# The VQGAN+CLIP (z+quantize method) notebook this was based on is by Katherine
# Crowson (https://github.com/crowsonkb

# Imports
import cv2
from CLIP import clip
from torchvision.transforms import functional as TF
from torchvision import transforms
from torch.nn import functional as F
from torch import nn, optim
from PIL import Image, ImageFile
from omegaconf import OmegaConf
import torch
import numpy as np
import kornia.augmentation as K

from base64 import b64encode

from pathlib import Path
import sys, random
import math
import srscaler
import time
import argparse
import pickle
import pyvirtualcam


sys.path.append('taming-transformers')
from taming.models import cond_transformer, vqgan



class vqgan:
    model_names = [
        "wikiart_16384"
        ]

    load_from_pickle = True
    current_model_index = 0

    model_names_full = ["faceshq",
                        "vqgan_imagenet_f16_16384",
                        "wikiart_16384",
                        "coco",
                        "drin_transformer",
                        "cin_transformer"]

    # @title Parámetros
    textos = ""
    channel = ""
    ancho = 256
    alto = 256
    video_fps = 60
    superres_factor = 3

    stepsize = 0.2
    imagen_inicial = "tdout_cam.jpg"  # @param {type:"string"}
    init_weight = 0.05
    max_iteraciones = 5  # @param {type:"number"}

    map_x = np.zeros((ancho, alto), dtype=np.float32)
    map_y = np.zeros((ancho, alto), dtype=np.float32)

    # @param ["vqgan_imagenet_f16_16384", "vqgan_imagenet_f16_1024", "wikiart_1024", "wikiart_16384", "coco", "faceshq", "sflckr", "ade20k", "ffhq", "celebahq", "gumbel_8192"]
    modelo = "vqgan_imagenet_f16_16384"
    intervalo_imagenes = 1  # @param {type:"number"}
    imagenes_objetivo = None  # @param {type:"string"}
    seed = 5  # @param {type:"number"}
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    input_images = ""

    i = 0

    nombres_modelos = {"vqgan_imagenet_f16_16384": 'ImageNet 16384', "vqgan_imagenet_f16_1024": "ImageNet 1024",
                       "wikiart_1024": "WikiArt 1024", "wikiart_16384": "WikiArt 16384", "coco": "COCO-Stuff", "faceshq": "FacesHQ", "sflckr": "S-FLCKR", "ade20k": "ADE20K", "ffhq": "FFHQ", "celebahq": "CelebA-HQ", "gumbel_8192": "Gumbel 8192"}
    nombre_modelo = nombres_modelos[modelo]

    verbs = ["spinning", "dreaming", "watering", "loving", "eating",
             "drinking", "sleeping", "repeating", "surreal", "psychedelic"]
    nouns = ["fish", "egg", "peacock", "watermelon", "pickle", "horse", "dog", "house", "kitchen", "bedroom", "door", "table", "lamp", "dresser", "watch", "logo", "icon", "tree",
             "grass", "flower", "plant", "shrub", "bloom", "screwdriver", "spanner", "figurine", "statue", "graveyard", "hotel", "bus", "train", "car", "lamp", "computer", "monitor"]

    frames = []
    frames_supeResed = []
    first_run = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_capability())

    def __init__(self) -> None:

        self.cam_init()
        self.res_scaler = srscaler.upscaler(cuda_on = True, factor = self.superres_factor)
        print(f" {bcolors.OKGREEN}[SRCNN] Image SuperRes device: {self.res_scaler.device}{bcolors.ENDC}")
        self.img_latest = 0

        if self.seed == -1:
            self.seed = None
        if self.imagen_inicial == "None":
            self.imagen_inicial = None

        if self.imagenes_objetivo == "None" or not self.imagenes_objetivo:
            self.imagenes_objetivo = []
        else:
            self.imagenes_objetivo = self.imagenes_objetivo.split("|")
            self.imagenes_objetivo = [image.strip()
                                      for image in self.imagenes_objetivo]

        if self.imagen_inicial or self.imagenes_objetivo != []:
            input_images = True

        self.textos = [frase.strip() for frase in self.textos.split("|")]
        if self.textos == ['']:
            self.textos = []

        self.args = argparse.Namespace(
            prompts=self.textos,
            image_prompts=self.imagenes_objetivo,
            noise_prompt_seeds=[],
            noise_prompt_weights=[],
            size=[self.ancho, self.alto],
            init_image=self.imagen_inicial,
            init_weight=self.init_weight,
            clip_model='ViT-B/32',
            vqgan_config=f'models/{self.modelo}.yaml',
            vqgan_checkpoint=f'models/{self.modelo}.ckpt',
            step_size=self.stepsize,
            cutn=8,
            cut_pow=1.,
            display_freq=self.intervalo_imagenes,
            seed=self.seed,
        )

        args = self.args

        print(f" {bcolors.OKBLUE}[CUDA] Using device: {self.device}{bcolors.ENDC}")
        if self.textos:
            print(' [CONFIG] Using texts:', self.textos)
        if self.imagenes_objetivo:
            print(' [CONFIG] Using image prompts:', self.imagenes_objetivo)
        if args.seed is None:
            seed = torch.seed()
        else:
            seed = args.seed

        torch.manual_seed(seed)
        print(' [CONFIG] Using seed:', seed)


        # Load pickle models
        if self.load_from_pickle:
            print(f" {bcolors.OKBLUE}[MODELS] Loading from local pickle... {bcolors.ENDC}")
            with open("pickle_models_1", "rb") as f:
                self.models = pickle.load(f)
        else:
            print(f" {bcolors.OKBLUE}[MODELS] Loading from models folder... {bcolors.ENDC}")
            self.models = [load_vqgan_model(
                f"models/{name}.yaml", f"models/{name}.ckpt").to(self.device) for name in self.model_names]

            with open(f"pickle_models_{len(self.models)}", "wb") as f:
                pickle.dump(self.models, f)
        print(f" {bcolors.OKBLUE}[MODELS] Successfully loaded {len(self.models)} models! {bcolors.ENDC}")


        self.model = self.models[0]
        self.perceptor = clip.load(args.clip_model, jit=False)[
            0].eval().requires_grad_(False).to(self.device)

        self.cut_size = self.perceptor.visual.input_resolution

        self.e_dim = self.model.quantize.e_dim

        self.f = 2**(self.model.decoder.num_resolutions - 1)
        self.make_cutouts = MakeCutouts(
            self.cut_size, args.cutn, cut_pow=args.cut_pow)

        n_toks = self.model.quantize.n_e

        toksX, toksY = args.size[0] // self.f, args.size[1] // self.f
        sideX, sideY = toksX * self.f, toksY * self.f

        self.z_min = self.model.quantize.embedding.weight.min(
            dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(
            dim=0).values[None, :, None, None]

        if args.init_image:
            pil_image = Image.open(
                args.init_image).convert('RGB')
            pil_image = pil_image.resize(
                (sideX, sideY), Image.LANCZOS)
            self.img_latest = pil_image
            self.z, *_ = self.model.encode(TF.to_tensor(
                pil_image).to(self.device).unsqueeze(0) * 2 - 1)

        else:
            one_hot = F.one_hot(torch.randint(
                n_toks, [toksY * toksX], device=self.device), n_toks).float()
            self.z = one_hot @ self.model.quantize.embedding.weight
            self.z = self.z.view(
                [-1, toksY, toksX, self.e_dim]).permute(0, 3, 1, 2)

        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)
        self.opt = optim.Adagrad([self.z], lr=args.step_size)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                  std=[0.26862954, 0.26130258, 0.27577711])


        self.pMs = []
        self.args.prompts = []
        #self.generate("/imagenet", args.prompts[0])

        print(f' {bcolors.OKGREEN}[STATUS] Init complete. {bcolors.ENDC} \n\n')
        print(f' {bcolors.OKBLUE}[MODELS] Current model:  {self.model_names[self.current_model_index]} ')

    def switch_model(self, model_index=0):
        self.current_model_index = model_index
        if self.model != self.models[model_index]:
            self.model = self.models[model_index]
            print(f' {bcolors.OKBLUE}[MODELS] switched to {self.model_names[model_index]}{bcolors.ENDC}')

        #model_index = random.randint(0,len(self.models)-1)
            


    def cam_init(self):
        self.cam = pyvirtualcam.Camera(
            width=self.ancho * self.superres_factor, height=self.alto * self.superres_factor, fps=30)
        print(
            f' {bcolors.OKBLUE}[DEVICE] Using virtual Camera: {self.cam.device} {bcolors.ENDC}')

    def synth(self):
        z_q = vector_quantize(self.z.movedim(
            1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    # @torch.inference_mode()
    def checkin(self, i, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        print(f' {bcolors.WARNING}[VQGAN] step: {i}/{self.max_iteraciones * ((i-1)//self.max_iteraciones + 1)}, loss: {sum(losses).item():g}, losses: {losses_str} {bcolors.ENDC}', end="\r")
        out = self.synth()

    def ascend_txt(self):
        out = self.synth()
        iii = self.perceptor.encode_image(
            self.normalize(self.make_cutouts(out))).float()

        result = []

        if self.args.init_weight:
            result.append(F.mse_loss(
                self.z, self.z_orig) * self.args.init_weight / 2)

        for prompt in self.pMs:
            result.append(prompt(iii))
            
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu(
        ).detach().numpy().astype(np.uint8))[:, :, :]
        img = np.transpose(img, (1, 2, 0))
        
        self.img_latest = img
        self.frames.append(img)
        #img_file = Image.fromarray(img, 'RGB')

        # =============================================
        # Upscale 3x SRCNN
        # =============================================

        img_3x = np.uint8((self.res_scaler.up(img)))
        self.frames_supeResed.append(img_3x)

        self.cam.send(img_3x)
        self.cam.sleep_until_next_frame()

        return result

    def train(self):
        self.i += 1
        self.opt.zero_grad(set_to_none=True)
        lossAll = self.ascend_txt()
        if self.i % self.args.display_freq == 0:
            self.checkin(self.i, lossAll)
        loss = sum(lossAll)
        loss.backward()

        self.opt.step()
        # with torch.no_grad():

        with torch.inference_mode():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))


    def refresh_z(self):

        self.pMs = []
        self.args.prompts = []
        
        # Load new init image for next round
        self.z, *_ = self.model.encode(TF.to_tensor(
            self.img_latest).to(self.device).unsqueeze(0) * 2 - 1)
        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)
        self.opt = optim.Adagrad(
            [self.z], lr=self.args.step_size)

        # start keyframe
        batch = self.make_cutouts(TF.to_tensor(
            self.img_latest).unsqueeze(0).to(self.device))
        embed = self.perceptor.encode_image(self.normalize(batch)).float()
         
        self.pMs.append(Prompt(
            embed, self.init_weight, float('-inf')).to(self.device))


    def generate(
            self,
            channel,
            input_prompt="",
            target_image="",
            pMs_size = 1,
            ramdisk=False):

        start = time.time()

        toksX, toksY = self.args.size[0] // self.f, self.args.size[1] // self.f
        sideX, sideY = toksX * self.f, toksY * self.f

        self.refresh_z()

        # Text Prompt
        if input_prompt:
            print(f" {bcolors.OKCYAN}[PROMPT] {input_prompt} {bcolors.ENDC}")
            self.args.prompts.append(input_prompt)

            txt, weight, stop = parse_prompt(input_prompt)
            embed = self.perceptor.encode_text(
                clip.tokenize(txt).to(self.device)).float()
            self.pMs.append(Prompt(
                embed, weight, stop).to(self.device))

        # Target Img
        if target_image:

            # target keyframe
            path, weight, stop = parse_prompt(target_image)
            if ramdisk:
                path = "R:/" + path

            # force weight
            weight = 1 - self.init_weight

            img = resize_image(
                Image.open(path).convert('RGB'), (sideX, sideY))

            batch = self.make_cutouts(TF.to_tensor(
                img).unsqueeze(0).to(self.device))
            embed = self.perceptor.encode_image(self.normalize(batch)).float()
            
            self.pMs.append(Prompt(
                embed, weight, stop).to(self.device))

            # Randomize Init Weight
            #self.args.init_weight = round(random.uniform(0.04, 0.06), 2)
            #self.args.step_size = round(random.uniform(0.4, 0.6), 2)

        print(f'\n')
        for iter in range(self.max_iteraciones):
            self.train()

        time_measure = round((time.time()-start), 2)
        fps = round((1 / time_measure * self.max_iteraciones), 2)
        print(f" \n {bcolors.OKGREEN}[METRIC] Execution time {time_measure} s, fps {fps}, upres_time {self.res_scaler.last_metric}s {bcolors.ENDC}", end="\r")


    # Video
    def save_video(self, video_name="default_mov_out", ramdisk=False, interp_frames=9):
        if video_name:
            video_name = video_name + "_" + \
                str(random.randint(0, 10000)) + ".mp4"
            if ramdisk:
                video_name = "R:/" + video_name

            writer = cv2.VideoWriter(
                video_name, 
                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                self.video_fps, 
                (self.alto * self.superres_factor, self.ancho * self.superres_factor))
            
            f_counter = 0

            # add interpolation to smooth out video
            for frame_index in range(len(self.frames_supeResed)-1):
                writer.write(self.frames_supeResed[frame_index])
                self.cam.send(np.uint8(self.frames_supeResed[frame_index]))
                self.cam.sleep_until_next_frame()

                f_counter += 1
                interp_frame = self.frames_supeResed[frame_index].copy()
                interp_step = 1 / (interp_frames + 1)
                for interp_index in range(interp_frames):
                    weight = interp_step * (interp_index + 1)
                    cv2.addWeighted(
                        self.frames_supeResed[frame_index], 1 - weight,
                        self.frames_supeResed[frame_index + 1], weight, 0.0,
                        dst=interp_frame)
                    writer.write(interp_frame)
                    self.cam.send(np.uint8(interp_frame))
                    self.cam.sleep_until_next_frame()
                    f_counter += 1

            writer.release()
            print(
                f"\n {bcolors.OKCYAN}[VIDEO] {f_counter} frames saved to {video_name}{bcolors.ENDC}")


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + \
        codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(
            dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    #print(f"\n[parse_prompt()] p {vals[0]} w {round(float(vals[1]), 2)} s {round(float(vals[2]), 2)}")
    return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.0),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1,
                           p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7))
        self.noise_fac = 0.1

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow *
                       (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety +
                           size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]
                                   ).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(
            **config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        print(config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)
