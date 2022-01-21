# Base VQGAN CLIP functions and helpers
# The VQGAN+CLIP (z+quantize method) notebook this was based on is by Katherine
# Crowson (https://github.com/crowsonkb
import os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
sys.path.insert(1, f'{ROOT_DIR}/taming-transformers')


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
from BColors import BColors

from pathlib import Path

import math

import time
import argparse
import pickle
import pyvirtualcam
import taming
from upscalers.upscaler import Upscaler

import sys, random



class VQGanClip:
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    args = []
    verbs = ["spinning", "dreaming", "watering", "loving", "eating",
             "drinking", "sleeping", "repeating", "surreal", "psychedelic"]
    nouns = ["fish", "egg", "peacock", "watermelon", "pickle", "horse", "dog", "house", "kitchen", "bedroom", "door", "table", "lamp", "dresser", "watch", "logo", "icon", "tree",
             "grass", "flower", "plant", "shrub", "bloom", "screwdriver", "spanner", "figurine", "statue", "graveyard", "hotel", "bus", "train", "car", "lamp", "computer", "monitor"]

    models = []
    frames = []
    frames_supeResed = []

    standby_last_index = 1

    first_run = True
    i = 0
    cam_last_frame = -1
    lastest_frame_push = False
    lastest_frame = None

    print(torch.cuda.get_device_capability())

    def __init__(self, args):
        self.args = args

        self.map_x = np.zeros((args.ancho, args.alto), dtype=np.float32)
        self.map_y = np.zeros((args.ancho, args.alto), dtype=np.float32)

        self.cam_init()
        self.res_scaler = Upscaler(cuda_on = True, factor = args.superres_factor)
        
        print(f" {BColors.OKGREEN}[SRCNN] Image SuperRes device: {self.res_scaler.device}{BColors.ENDC}")
        self.img_latest = 0

        if args.seed == -1:
            args.seed = None

        print(f" {BColors.OKBLUE}[CUDA] Using device: {args.device}{BColors.ENDC}")

        torch.manual_seed(args.seed)
        print(' [CONFIG] Using seed:', args.seed)


        # -> Load pickle models
        if args.load_from_pickle:
            print(f" {BColors.OKBLUE}[MODELS] Loading from local pickle... {BColors.ENDC}")
            with open(f"{ROOT_DIR}/pickles/pickle_models_{args.model_names[args.current_model_index]}", "rb") as f:
                self.models = pickle.load(f)
                
        else:
            print(f" {BColors.OKBLUE}[MODELS] Loading from models folder... {BColors.ENDC}")
            self.models = [load_vqgan_model(
                f"{ROOT_DIR}/models/{name}.yaml", f"models/{name}.ckpt").to(args.device) for name in args.model_names]

            with open(f"{ROOT_DIR}/pickles/pickle_models_{args.model_names[args.current_model_index]}", "wb") as f:
                pickle.dump(self.models, f)

        print(f" {BColors.OKBLUE}[MODELS] Successfully loaded {len(self.models)} models! {BColors.ENDC}")

        

        # -> Init Frame
        self.model = self.models[args.current_model_index]
        self.perceptor = clip.load(args.clip_model, jit=False)[
            0].eval().requires_grad_(False).to(args.device)

        self.cut_size = self.perceptor.visual.input_resolution

        self.e_dim = self.model.quantize.e_dim

        self.f = 2**(self.model.decoder.num_resolutions - 1)
        self.make_cutouts = MakeCutouts(
            self.cut_size, args.cutn, cut_pow=args.cut_pow)

        n_toks = self.model.quantize.n_e

        toksX, toksY = args.ancho // self.f, args.alto // self.f
        sideX, sideY = toksX * self.f, toksY * self.f

        self.z_min = self.model.quantize.embedding.weight.min(
            dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(
            dim=0).values[None, :, None, None]

        if args.init_image:
            pil_image = Image.open(
                f"{ROOT_DIR}/images/{args.init_image}").convert('RGB')
            pil_image = pil_image.resize(
                (sideX, sideY), Image.LANCZOS)
            self.img_latest = pil_image
            self.z, *_ = self.model.encode(TF.to_tensor(
                pil_image).to(args.device).unsqueeze(0) * 2 - 1)

        else:
            one_hot = F.one_hot(torch.randint(
                n_toks, [toksY * toksX], device=args.device), n_toks).float()
            self.z = one_hot @ self.model.quantize.embedding.weight
            self.z = self.z.view(
                [-1, toksY, toksX, self.e_dim]).permute(0, 3, 1, 2)

        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)
        self.opt = optim.Adagrad([self.z], lr=args.step_size)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                  std=[0.26862954, 0.26130258, 0.27577711])


        self.pMs = []
        #args.prompts = []
        #self.generate("/imagenet", args.prompts[0])

        print(f' {BColors.OKGREEN}[STATUS] Init complete. {BColors.ENDC} \n\n')
        print(f' {BColors.OKBLUE}[MODELS] Current model:  {args.model_names[args.current_model_index]} ')

    def switch_model(self, model_index=0):
        self.args.current_model_index = model_index
        if self.model != self.models[model_index]:
            self.model = self.models[model_index]
            print(f' {BColors.OKBLUE}[MODELS] switched to {self.args.model_names[model_index]}{BColors.ENDC}')

        #model_index = random.randint(0,len(self.models)-1)
            


    def cam_init(self):
        args = self.args
        self.cam = pyvirtualcam.Camera(
            width = args.ancho * args.superres_factor, 
            height = args.alto * args.superres_factor, 
            fps = args.video_fps)
        
        print(
            f' {BColors.OKBLUE}[DEVICE] Using virtual Camera: {self.cam.device} {BColors.ENDC}')

    def synth(self):
        z_q = vector_quantize(self.z.movedim(
            1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    # @torch.inference_mode()
    def checkin(self, i, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        print(f' {BColors.WARNING}[VQGAN]  step: {i}/{self.args.max_iteraciones * ((i-1)//self.args.max_iteraciones + 1)}, loss: {sum(losses).item():g}, losses: {losses_str} {BColors.ENDC}', end="\r")
        #out = self.synth()

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
        
        self.img_latest = Image.fromarray(img)
        self.frames.append(img)

        # =============================================
        # Upscale 4x SRCNN
        # =============================================

        img_nx = np.uint8((self.res_scaler.up(img)))
        self.frames_supeResed.append(img_nx)
        self.lastest_frame_push = True
        self.lastest_frame = np.copy(img_nx)

        # Maintain buffer frames
        buffer_frames = 10
        if len(self.frames_supeResed) > buffer_frames:
            self.frames_supeResed.pop(0)

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
        with torch.inference_mode():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))


    def refresh_z(self, style_transfer=""):
        args = self.args
        self.pMs = []
        if self.i > 0:
            self.prompts = [] 
        
        #  <- FORCE INJECT STYLE from image
        if style_transfer:
            pil_image = Image.open(
                f"R:/{style_transfer}").convert('RGB')
            pil_image = pil_image.resize(
                (args.ancho, args.alto), Image.LANCZOS)
            #print(pil_image, self.img_latest)
            self.img_latest = Image.blend(self.img_latest, pil_image, alpha=.25)

        # Load new init image for next round
        self.z, *_ = self.model.encode(TF.to_tensor(
            self.img_latest).to(args.device).unsqueeze(0) * 2 - 1)
        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)
        self.opt = optim.Adagrad(
            [self.z], lr=args.step_size)

        # start keyframe
        batch = self.make_cutouts(TF.to_tensor(
            self.img_latest).unsqueeze(0).to(args.device))
        embed = self.perceptor.encode_image(self.normalize(batch)).float()
         
        self.pMs.append(Prompt(
            embed, args.init_weight, float('-inf')).to(args.device))





    def interp_cam_step(self, buffer_range=3):
        # interpolate from previous frame to newest frame
        if len(self.frames_supeResed) >= buffer_range:
            if self.first_run:
                self.last_output_frame = np.copy(self.frames_supeResed[-2])
                self.first_run = False

            # select a new target frame but not previous one
            choices = [i for i in range(1, buffer_range)]
            choices.remove(self.standby_last_index)
            i_rand = random.choice(choices)
            self.standby_last_index = i_rand

            if self.lastest_frame_push:
                newest_frame_snapshot = np.copy(self.lastest_frame)
                self.interp_out(
                    prev_frame=np.copy(self.last_output_frame), 
                    newest_frame=newest_frame_snapshot)
                self.last_output_frame = newest_frame_snapshot
                self.lastest_frame_push = False
            else:
                newest_frame_snapshot = np.copy(self.frames_supeResed[-i_rand])
                self.interp_out(
                    prev_frame=np.copy(self.last_output_frame), 
                    newest_frame=newest_frame_snapshot)
                self.last_output_frame = newest_frame_snapshot



    def interp_out(self, prev_frame, newest_frame, interp_framerate=15):
        # Output interpolated / smoothed frames between newest frame and previous frame
        wait_sync = 1 / interp_framerate
        interp_frame = np.copy(prev_frame)
        interp_step = 1 / (interp_framerate)
        

        for interp_index in range(interp_framerate):
            
            sync_start = time.time()
            weight = interp_step * interp_index
            cv2.addWeighted(
                prev_frame, 1 - weight,
                newest_frame, weight, 0.0,
                dst=interp_frame)

            self.cam.send(np.uint8(interp_frame))
            self.cam.sleep_until_next_frame()
            time.sleep(wait_sync - (time.time() - sync_start))
        


    def generate(
            self,
            channel,
            input_prompt="",
            target_image="",
            ramdisk=False,
            optimize_steps=-1,
            style_transfer="",
            target_weight=1,
            step_size=0):

        start = time.time()
        args = self.args
        toksX, toksY = args.ancho // self.f, args.alto // self.f
        sideX, sideY = toksX * self.f, toksY * self.f

        # Big steps for manual inputs
        if step_size != 0:
            args.step_size = step_size
        else:
            args.step_size = 0.2 # standby step size

        self.refresh_z(style_transfer=style_transfer)


        # Text Prompt
        if input_prompt:
            if style_transfer:
                print(f" {BColors.OKCYAN}[STYLE]  {input_prompt} {BColors.ENDC}")
            else:
                print(f" {BColors.OKCYAN}[PROMPT]{input_prompt} {BColors.ENDC}")
            self.args.prompts.append(input_prompt)

            txt, weight, stop = parse_prompt(input_prompt)
            embed = self.perceptor.encode_text(
                clip.tokenize(txt).to(args.device)).float()

            # force weight
            self.pMs.append(Prompt(
                embed, weight, stop).to(args.device))


        # Target Img
        if target_image:

            # target keyframe
            path, weight, stop = parse_prompt(target_image)

            if ramdisk:
                path = args.ramdisk + path
            else:
                path = ROOT_DIR + "/" + path

            # force weight
            # weight = 1 - args.init_weight

            try:
                img = resize_image(
                    Image.open(path).convert('RGB'), (sideX, sideY))
            except Exception:
                img = self.img_latest.copy()
                print(path)
                print(f" {BColors.FAIL}[READ]   Corrupted image input. Skipping frame. {BColors.ENDC}")
                return

            batch = self.make_cutouts(TF.to_tensor(
                img).unsqueeze(0).to(args.device))
            embed = self.perceptor.encode_image(self.normalize(batch)).float()
            
            self.pMs.append(Prompt(
                embed, weight, stop).to(args.device))



        # sync training step to seconds
        if optimize_steps == -1:
            optimize_steps = args.max_iteraciones

        for _ in range(optimize_steps):
            sync_start = time.time()
            self.train()
            time.sleep(1 - (time.time() % 1))

        time_measure = round((time.time()-start), 2)
        fps = round((1 / time_measure * args.max_iteraciones), 2)
        print(f" \n {BColors.OKGREEN}[METRIC] Execution time {time_measure} s, fps {fps}, upres_time {self.res_scaler.last_metric}s {BColors.ENDC}")


    # Video
    def save_video(self, video_name="default_mov_out"):
        args = self.args
        if video_name:
            video_name = video_name + "_" + \
                str(random.randint(0, 10000)) + ".mp4"

            if args.ramdisk:
                video_name = args.ramdisk + video_name
            else:
                video_name = ROOT_DIR + "/" + video_name

            writer = cv2.VideoWriter(
                video_name, 
                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                args.video_fps, 
                (args.alto * args.superres_factor, 
                args.ancho * args.superres_factor))
            
            f_counter = 0

            # -> add interpolation to smooth out video
            for frame_index in range(len(self.frames_supeResed)-1):
                writer.write(self.frames_supeResed[frame_index])
                self.cam.send(np.uint8(self.frames_supeResed[frame_index]))
                self.cam.sleep_until_next_frame()

                f_counter += 1
                interp_frame = self.frames_supeResed[frame_index].copy()
                interp_step = 1 / (args.video_interp_frames + 1)
                for interp_index in range(args.video_interp_frames):
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
                f"\n {BColors.OKCYAN}[VIDEO] {f_counter} frames saved to {video_name}{BColors.ENDC}")



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
            K.RandomSolarize(0.01, 0.01, p=0.7),
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
        model = taming.models.vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = taming.models.cond_transformer.Net2NetTransformer(
            **config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = taming.models.vqgan.GumbelVQ(**config.model.params)
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
