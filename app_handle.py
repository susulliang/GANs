# import support functions and classes

import traceback

import numpy as np
import torch.multiprocessing as mp

import vqgan_helper
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import OSCUDPServer

from srcnn_scaler import upscaler
from srcnn_models.model import SRCNN

vqgan = vqgan_helper.vqgan
bcolors = vqgan_helper.bcolors


class osc_handle:
    # -> Command Line Handler 
    def __init__(self, vq_ref, ip="192.168.0.138", port=5006) -> None:
        self.vq = vq_ref
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self.default_handler)
        server = OSCUDPServer((ip, port), dispatcher)

        print('\n\n [OSC] Listening for OSC data on ' +
              ip + " at port " + str(port))
        server.serve_forever()  # Blocks forever

    def print_handler(self, address, *osc_args):
        print(f"{address}: {osc_args}")

    def default_handler(self, address, *osc_args):
        channel = str(osc_args[0]).strip()
        input_prompt = str(osc_args[1]).strip()

        if channel not in ["/wikiart", "/prompt", "/coco", "/imagenet", "/color", "/noun"]:
            exit()

        if input_prompt.isnumeric():
            exit()

        if input_prompt.strip() == "":
            exit()

        self.vq.generate(channel, input_prompt)


class command_handle:
    # -> Command Line Handler 

    def __init__(self, vq_ref) -> None:
        self.vq = vq_ref
        pass

    def loop(self):
        while True:
            channel, input_prompt = self.process_input(
                input("Your prompt here: "))
            self.vq.generate(channel, input_prompt)

    def process_input(input_prompt):
        input_prompt = input_prompt.strip()
        return "/wikiart", input_prompt

    def test_prompts(self):
        print(" [DEBUG] Using default test prompt sequence")
        prompts = "cube / spiral / ocean and beach / night and moon / dark sky and moon / green and orange colors / broccoli and vegetable"
        prompts = "winter day with some trees / a snowy russian forest / forest with very beautiful landscape"
        prompts = prompts.split(" / ")
        for input_prompt in prompts:
            self.vq.generate("/imagenet", input_prompt)

    def test_image_prompts(self, max_iter=20):
        target_image_file = "tdout_noise.jpg"
        print(f" [DEBUG] Using {target_image_file} as target image")

        for i in range(max_iter):
            self.vq.generate(
                channel="/imagenet",
                target_image=target_image_file,
                ramdisk=True)
                

        self.vq.save_video(
            video_name="vid_interp_out",
            ramdisk=True)


def main():
    vq = vqgan()
    handle = command_handle(vq)
    handle.test_prompts()

    
    print(f' {bcolors.OKGREEN}[STATUS] Command completed, graceful byebye! {bcolors.ENDC} \n')

    # OSC MODE
    #handle = osc_handle()



if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()


# cube / spiral / ocean and beach / night and moon / dark sky and moon / green and orange colors / broccoli and vegetable
