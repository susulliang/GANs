import sys
from BColors import BColors
sys.path.append('.srcnn_models')
sys.path.append('.taming_transformers')

from argparse import Namespace


import ClassVQGan
import ClassSRCnn


import traceback

from ModeOSC import OSCHandle
from ModeCmd import CmdHandle
from model import SRCNN

import torch


def main():
    
    print(p for p in sys.path)
    my_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_names_full = ("faceshq",
                            "vqgan_imagenet_f16_16384",
                            "wikiart_16384",
                            "coco",
                            "drin_transformer",
                            "cin_transformer")
                            
    # -> Run Parameters
    fuckme = Namespace(a=1, b='c')
    print(fuckme)
    vq_args = Namespace(model_names="wikiart_16384",
        load_from_pickle=False,
        current_model_index=0,
        textos="",
        channel="",
        ancho=256,
        alto=256,
        video_fps=60,
        superres_factor=3,
        step_size=0.3,
        init_image="winter-forest.jpg",  # @param {type:"string"}
        init_weight=0.05,
        max_iteraciones=25,  # @param {type:"number"}
        seed=5,  # @param {type:"number"}
        prompts="",
        image_prompts="",
        noise_prompt_seeds="",
        noise_prompt_weights="",
        size="(256, 256)",
        clip_model='ViT-B/32',
        cutn=8,
        cut_pow=1.,
        display_freq=1,
        device=my_device,
        ramdisk="R:/",
        video_name="default_mov_out",
        video_interp_frames=9)

    # Global variables for storing networks, models and tensors
    vq = ClassVQGan.VQGanClip(args = vq_args)

    # Command Prompt Mode
    handle = CmdHandle(vq_ref = vq)
    handle.start_looping()

    # OSC Mode
    handle = OSCHandle(vq_ref = vq)
    handle.start_listening()
    print(f' {BColors.OKGREEN}[STATUS] Command completed, graceful byebye! {bcolors.ENDC} \n')



if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit()