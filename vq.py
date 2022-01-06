# import support functions and classes
import argparse
import random

import cv2
import imageio
import numpy as np
import pyvirtualcam
import torch
from PIL import Image, ImageFile
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

import gan_base_module


class vqgan:
    model_names = [ "coco",
                    "wikiart_16384", 
                    "vqgan_imagenet_f16_16384",
                    "drin_transformer",
                    "cin_transformer"]

    #@title Parámetros
    textos = "moon and dark sky"
    ancho =  256
    alto =  256
    stepsize = 0.25

    #@param ["vqgan_imagenet_f16_16384", "vqgan_imagenet_f16_1024", "wikiart_1024", "wikiart_16384", "coco", "faceshq", "sflckr", "ade20k", "ffhq", "celebahq", "gumbel_8192"]
    modelo = "vqgan_imagenet_f16_16384" 

    #modelo = "wikiart_16384" 
    intervalo_imagenes = 1 #@param {type:"number"}
    imagen_inicial = "xi.jpg" #@param {type:"string"}
    imagenes_objetivo = None #@param {type:"string"}
    seed = 5 #@param {type:"number"}
    max_iteraciones = 25 #@param {type:"number"}
    input_images = ""

    i = 0

    nombres_modelos={"vqgan_imagenet_f16_16384": 'ImageNet 16384',"vqgan_imagenet_f16_1024":"ImageNet 1024", 
                    "wikiart_1024":"WikiArt 1024", "wikiart_16384":"WikiArt 16384", "coco":"COCO-Stuff", "faceshq":"FacesHQ", "sflckr":"S-FLCKR", "ade20k":"ADE20K", "ffhq":"FFHQ", "celebahq":"CelebA-HQ", "gumbel_8192": "Gumbel 8192"}
    nombre_modelo = nombres_modelos[modelo]     

    verbs = ["spinning","dreaming","watering","loving","eating","drinking","sleeping","repeating","surreal","psychedelic"]
    nouns = ["fish","egg","peacock","watermelon","pickle","horse","dog","house","kitchen","bedroom","door","table","lamp","dresser","watch","logo","icon","tree",
    "grass","flower","plant","shrub","bloom","screwdriver","spanner","figurine","statue","graveyard","hotel","bus","train","car","lamp","computer","monitor"]

    frames = []
    first_run = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



    def __init__(self) -> None:

        self.cam_init()

        self.img_latest = 0
        
        if self.seed == -1:
            self.seed = None
        if self.imagen_inicial == "None":
            self.imagen_inicial = None


        if self.imagenes_objetivo == "None" or not self.imagenes_objetivo:
            self.imagenes_objetivo = []
        else:
            self.imagenes_objetivo = self.imagenes_objetivo.split("|")
            self.imagenes_objetivo = [image.strip() for image in self.imagenes_objetivo]

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
            init_weight=0.5,
            clip_model='ViT-B/32',
            vqgan_config=f'models/{self.modelo}.yaml',
            vqgan_checkpoint=f'models/{self.modelo}.ckpt',
            step_size=self.stepsize,
            cutn=16,
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

        

        self.models = [gan_base_module.load_vqgan_model(f"models/{name}.yaml", f"models/{name}.ckpt").to(self.device) for name in self.model_names]

        self.model = self.models[0]
        self.perceptor = gan_base_module.clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(self.device)

        self.cut_size = self.perceptor.visual.input_resolution

        self.e_dim = self.model.quantize.e_dim

        self.f = 2**(self.model.decoder.num_resolutions - 1)
        self.make_cutouts = gan_base_module.MakeCutouts(self.cut_size, args.cutn, cut_pow=args.cut_pow)

        n_toks = self.model.quantize.n_e

        toksX, toksY = args.size[0] // self.f, args.size[1] // self.f
        sideX, sideY = toksX * self.f, toksY * self.f

        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        if args.init_image:
            pil_image = gan_base_module.Image.open(args.init_image).convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), gan_base_module.Image.LANCZOS)
            self.img_latest = pil_image
            self.z, *_ = self.model.encode(gan_base_module.TF.to_tensor(pil_image).to(self.device).unsqueeze(0) * 2 - 1)
 
        else:
            one_hot = gan_base_module.F.one_hot(torch.randint(n_toks, [toksY * toksX], device=self.device), n_toks).float()
            self.z = one_hot @ self.model.quantize.embedding.weight
            self.z = self.z.view([-1, toksY, toksX, self.e_dim]).permute(0, 3, 1, 2)

        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)
        self.opt = gan_base_module.optim.Adagrad([self.z], lr=args.step_size)
        self.normalize = gan_base_module.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])

        self.pMs = []
        self.generate("/coco", args.prompts[0])

        print(f' {bcolors.OKGREEN}[STATUS] Init complete. {bcolors.ENDC} \n\n\n\n')

    def switch_model(self, model_index = 0):
        #model_index = random.randint(0,len(self.models)-1)
        self.model = self.models[model_index]
        print(f' [MODEL] switched to {self.model_names[model_index]}')
        self.z_orig = self.z.clone()
        self.z, *_ = self.model.encode(gan_base_module.TF.to_tensor(self.img_latest).to(self.device).unsqueeze(0) * 2 - 1)
        self.z.requires_grad_(True)
        self.opt = gan_base_module.optim.Adagrad([self.z], lr=self.args.step_size)
        #self.opt = self.opt2

    def cam_init(self):
        self.cam = pyvirtualcam.Camera(width=self.ancho, height=self.alto, fps=30)
        print(f' {bcolors.OKBLUE}[DEVICE] Using virtual Camera: {self.cam.device} {bcolors.ENDC}')

    def synth(self):
        z_q = gan_base_module.vector_quantize(self.z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return gan_base_module.clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    #@torch.no_grad()
    @torch.inference_mode()

    def checkin(self, i, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        print(f' {bcolors.WARNING}[GENERATE] step: {i}/{self.max_iteraciones * ((i-1)//self.max_iteraciones + 1)}, loss: {sum(losses).item():g}, losses: {losses_str} {bcolors.ENDC}', end = "\r")
        out = self.synth()

    def ascend_txt(self):
        out = self.synth()
        iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()

        result = []

        if self.args.init_weight:
            result.append(gan_base_module.F.mse_loss(self.z, self.z_orig) * self.args.init_weight / 2)

        for prompt in self.pMs:
            result.append(prompt(iii))
        

        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
        img = np.transpose(img, (1, 2, 0))
        self.img_latest = img
        self.frames.append(img)

        filename = f"steps_{self.i:04}.png"
        img_file = Image.fromarray(img, 'RGB')
        img_file.save("lastest.png")

        self.cam.send(np.uint8(img_file))
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
        #with torch.no_grad():

        with torch.inference_mode():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))


    def generate(self, channel, input_prompt):

        self.args.prompts.append(input_prompt)
        self.pMs = []
        toksX, toksY = self.args.size[0] // self.f, self.args.size[1] // self.f
        sideX, sideY = toksX * self.f, toksY * self.f
        
        # Text Prompt
        txt, weight, stop = gan_base_module.parse_prompt(input_prompt)
        embed = self.perceptor.encode_text(gan_base_module.clip.tokenize(txt).to(self.device)).float()
        self.pMs.append(gan_base_module.Prompt(embed, weight, stop).to(self.device))

        # Target Img
        path, weight, stop = gan_base_module.parse_prompt("dezzy.jpg")
        img = gan_base_module.resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
        batch = self.make_cutouts(gan_base_module.TF.to_tensor(img).unsqueeze(0).to(self.device))
        embed = self.perceptor.encode_image(self.normalize(batch)).float()
        self.pMs.append(gan_base_module.Prompt(embed, weight, stop).to(self.device))


        if channel == "/coco":
            self.switch_model(0)
        if channel == "/wikiart":
            self.switch_model(1)
        if channel == "/imagenet":
            self.switch_model(2)

        print(f" {bcolors.OKCYAN}[PROMPT] {input_prompt} {bcolors.ENDC}")

        for iter in range(self.max_iteraciones):
            self.train()

        # Video
        writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 60, (256,256))
        for frame in self.frames:
            writer.write(frame)
        writer.release()
        print(f'\n')

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



# == OSC handlers
class osc_handle:

    def __init__(self, ip = "192.168.0.138", port = 5006) -> None:

        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self.default_handler)
        server = BlockingOSCUDPServer((ip, port), dispatcher)

        print('\n\n\n\n [OSC] Listening for OSC data on ' + ip + " at port " + str(port))
        server.serve_forever()  # Blocks forever


    def print_handler(address, *osc_args):
        print(f"{address}: {osc_args}")

    def default_handler(address, *osc_args):
        global vq

        channel = str(osc_args[0]).strip()
        input_prompt = str(osc_args[1]).strip()

        if channel not in ["/wikiart", "/prompt", "/coco", "/imagenet", "/color", "/noun"]:
            exit()

        if input_prompt.isnumeric():
            exit() 

        if input_prompt.strip() == "":
            exit()

        vq.generate(channel, input_prompt)



    
def main():
    o = osc_handle()


vq = vqgan()
if __name__ == "__main__":
    try:
        main()
    except:
        exit()












# cube / spiral / ocean and beach / night and moon / dark sky and moon / green and orange colors / broccoli and vegetable
