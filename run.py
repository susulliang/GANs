import sys, os, traceback

# -------------------------------------
# -> ROOT_DIR should be on realtime-gan
# -------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
print(f'Launching from {ROOT_DIR}')
sys.path.insert(0,f'{ROOT_DIR}/upscalers')
# -------------------------------------






from argparse import Namespace
from BColors import BColors

import gan_VQGanClip

from handle_osc import OSCHandle
from handle_cmd import CmdHandle


import torch


def main():

    my_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_names_full = ("faceshq",
                            "vqgan_imagenet_f16_16384",
                            "wikiart_16384",
                            "coco",
                            "drin_transformer",
                            "cin_transformer",
                            "sflckr")
    
    tracking_img = "tdout_cam.jpg"

    # --------------------                        
    # -> Run Parameters
    # --------------------    
    vq_args = Namespace(model_names=["wikiart_16384"],
        model_names_full=model_names_full,
        load_from_pickle=True,
        current_model_index=0,
        textos="",
        channel="",
        ancho=256,
        alto=256,
        video_fps=60,
        superres_factor=3,
        step_size=0.15,
        init_image=tracking_img,  
        init_weight=0.05,
        max_iteraciones=60,  # @param {type:"number"}
        seed=5,  # @param {type:"number"}
        prompts=[],
        image_prompts="",
        size="(256, 256)",
        clip_model='ViT-B/32',
        cutn=12,
        cut_pow=1.,
        display_freq=1,
        device=my_device,
        ramdisk="R:/",
        video_name="default_mov_out",
        video_interp_frames=9)

    # Global variables for storing networks, models and tensors
    vq = gan_VQGanClip.VQGanClip(args=vq_args)

    # -> Automated Command Prompt Mode
    handle = CmdHandle(vq_ref=vq)
    handle.use_mic_inputs()

    handle.test_image_prompts(target_image_file="tdout_noise.jpg", max_iter =50)


    # -> OSC Mode
    handle = OSCHandle(vq_ref=vq)
    handle.start_listening()
    
    print(f' {BColors.OKGREEN}[STATUS] Command completed, graceful byebye! {BColors.ENDC} \n')



if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit()