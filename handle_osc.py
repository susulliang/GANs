# import support functions and classes

import sys

import numpy as np
import torch.multiprocessing as mp

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import OSCUDPServer

#from ClassSRCnn import Upscaler
#from model import SRCNN


class OSCHandle:
    # -> OSC SERVER Handler 
    def __init__(self, vq_ref, ip="192.168.0.138", port=5006) -> None:
        self.vq = vq_ref
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(handler = self.default_handler)
        self.server = OSCUDPServer((ip, port), dispatcher)

        print('\n\n [OSC] Listening for OSC data on ' +
              ip + " at port " + str(port))

    def start_listening(self):
        self.server.serve_forever()  # listen forever

    def default_handler(self, address = "", *osc_args):
        print('\n received osc packet')
        if len(osc_args) < 2:
            exit()

        channel = str(osc_args[0]).strip()
        input_prompt = str(osc_args[1]).strip()

        if channel not in ["/wikiart", "/prompt", "/coco", "/imagenet", "/color", "/noun"]:
            exit()

        if input_prompt.isnumeric():
            exit()

        if input_prompt.strip() == "":
            exit()

        self.vq.generate(channel, input_prompt)



# cube / spiral / ocean and beach / night and moon / dark sky and moon / green and orange colors / broccoli and vegetable
