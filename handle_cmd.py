import cmd, random
from pickletools import optimize
from queue import Empty
from numpy import empty
import speech_recognition as sr
from googletrans import Translator, constants
import time
import traceback
import _thread
from BColors import BColors
import glob

class CmdHandle:
    styles = ["fish eye view", "dreaming", "Kandinsy", "Cubist", "Abstract",
             "cartoon", "watercolor", "pixel", "pixelated", "simple", "pointillism", "acrylic", 
             "Ghost in the Shell style", "Death Stranding style", "gouache", "vignetting",
             "Dark and clean background", "small and center", "geometric", "linear", "Junji Ito style"]

    # -> Command Line Handler 
    def __init__(self, vq_ref) -> None:
        self.vq = vq_ref
        self.newest_prompt = ""
        self.mode = "standby"
        self.user_last_active = 0
        self.active_timeout = 15
        pass

    def start_looping(self):
        while True:
            channel, input_prompt = self.process_input(
                input("Your prompt here: "))
            self.vq.generate(channel, input_prompt)

    def process_input(input_prompt):
        input_prompt = input_prompt.strip()
        return "/wikiart", input_prompt

    def global_refresh(self, delay):
        print(
            f' {BColors.OKBLUE}[CANVAS] Ouput Framerate is set to {delay} FPS {BColors.ENDC}')
        c = 0
        while c < 500:
            sync_start = time.time()
            self.vq.interp_cam_step()
            # -> Force sync
            if delay - (time.time() - sync_start) > 0:
                time.sleep(delay - (time.time() - sync_start))
            
            

    def get_prompt(self):
        # -> Get user prompt from terminal
        while True:
            try:
                self.newest_prompt = input().strip()
            except Exception:
                self.newest_prompt = ""


    def exhibition(self, framerate=1):
        print(f' {BColors.OKBLUE}[MODE] Exhibition Mode {BColors.ENDC}')

        # -> Global Output Framerate Lock
        _thread.start_new_thread(self.get_prompt, ())
        _thread.start_new_thread(self.global_refresh, (1,))

        c = 0
        while True:
            # -> Global Loop
            if self.newest_prompt == "exit":
                exit()

            if self.newest_prompt != "" and self.mode == "standby":
                # -> Active Prompt Mode
                self.user_last_active = c
                _thread.start_new_thread(self.process_prompt, (self.newest_prompt,))
                self.newest_prompt = ""
                #self.vq.refresh_cam()

            elif self.mode == "standby":
                # -> Stanby Mode without user attention, trigger random generations
                if c - self.user_last_active > self.active_timeout:
                    print(f' {BColors.HEADER}[STATUS] Gathering ideas... {BColors.ENDC}', end='\r')
                    rand_choice_prob = random.randint(1, 100)
                    if rand_choice_prob < 10: 
                        # -> 10% Random Promopt
                        print(f' {BColors.HEADER}[STATUS] idea gathered from list of painting names {BColors.ENDC}')
                        _thread.start_new_thread(self.test_random_prompt, ())

                    elif rand_choice_prob < 15: 
                        # -> 5% Get Camera
                        print(f' {BColors.HEADER}[STATUS] idea gathered from camera input {BColors.ENDC}')
                        _thread.start_new_thread(self.test_cam_styletransfer, (2,))

                    elif rand_choice_prob < 25: 
                        # -> 10% Get Local Image Prompt
                        print(f' {BColors.HEADER}[STATUS] idea gathered from local images {BColors.ENDC}')
                        _thread.start_new_thread(self.test_image_input, ())
                else:
                    # -> Stanby Mode, with user attention
                    rand_choice_prob = random.randint(1, 100)
                    if rand_choice_prob < 40: 
                        # -> 40% Get Camera
                        print(f' {BColors.HEADER}[STATUS] idea gathered from camera input {BColors.ENDC}')
                        _thread.start_new_thread(self.test_cam_styletransfer, (5,))
                pass
                
                #_thread.start_new_thread(self.standby, ())exit

            # --------------
            c += 1
            
            time.sleep(1)


    def process_prompt(self, prompt):
        self.mode = "generating"
        self.vq.generate("/wikiart", prompt, step_size=0.6, optimize_steps=random.randint(20,40))
        self.mode = "standby"

    def test_random_prompt(self):
        self.mode = "generating"
        # Read prompts from local samples
        txt_file = open('prompts.txt', 'r')
        lines = txt_file.readlines()
        txt_file.close()

        # Strips the newline character
        random_line = random.choice(lines)
        parse = random_line.strip().split('.')
        self.vq.generate(
            "/imagenet", 
            input_prompt=parse[1], 
            step_size=0.3,
            optimize_steps=random.randint(10,25))

        self.mode = "standby"

    def test_image_input(self):
        self.mode = "generating"

        # -> Randomly load image from images folder
        self.vq.generate(
            channel="/wikiart",
            target_image=random.choice(glob.glob("images/*")),
            step_size=0.4,
            ramdisk=False,
            optimize_steps=random.randint(5,20))

        self.mode = "standby"

    def test_cam_styletransfer(self, transfer_iterations=1):
        self.mode = "generating"

        # -> Randomly load styles from predefined styles
        for _ in range(transfer_iterations):
            self.vq.generate(
                channel="/wikiart",
                style_transfer="tdout_cam.jpg",
                input_prompt=random.choice(self.styles),
                ramdisk=True,
                optimize_steps=random.randint(5,10))

        self.mode = "standby"


    def test_prompts(self, delay=1):
        print(" [DEBUG] Using default test prompt sequence")



        self.vq.save_video(video_name="vid_interp_out")


    def test_cmd_inputs(self, delay=1):
        print(" [DEBUG] Using default test prompt sequence")
        prompts = "cube / spiral / ocean and beach / night and moon / dark sky and moon / green and orange colors / broccoli and vegetable"
        prompts = "wassily kandinsky style landscapes / wassily kandinsky style cubes / beautiful sky"
        
        
        # Global Output Framerate Lock
        _thread.start_new_thread(self.global_clock, (delay,))

        cmd_input = "a beautiful dinner party with lots of apples"
        while cmd_input != "":
            self.vq.generate("/imagenet", cmd_input)
            cmd_input = input(f' {BColors.HEADER} Give me some ideas: {BColors.ENDC}').strip()
        
        self.vq.save_video(video_name="vid_interp_out")

    def test_image_prompts(
        self, 
        max_iter=100,
        target_image_file = "tdout_noise.jpg"):
        
        print(f" [DEBUG] Using {target_image_file} as target image")

        for i in range(max_iter):
            self.vq.generate(
                channel="/imagenet",
                target_image=target_image_file,
                ramdisk=True)
                

        self.vq.save_video(video_name="vid_interp_out")

    def use_mic_inputs(self, safe_word="apple", language="ru"):
        # Initialize recognizer class (for recognizing the speech)
        r = sr.Recognizer()
        t = Translator()
        # Reading Microphone as source
        # listening the speech and store in audio_text variable

        
        print("Waiting for mic prompts...")
        while True:
            try:
                with sr.Microphone() as source:
                    # -> listen for prompts
                    r.non_speaking_duration = 0.5
                    r.pause_threshold = 0.6
                    audio_data = r.listen(source, phrase_time_limit=5, timeout=10)
                    #speech = r.recognize_sphinx(audio_data)
                    speech_g = r.recognize_google(audio_data, language=language)
                    translate_g = t.translate(speech_g)
                    #print(speech)
                    print(speech_g)
                    print(translate_g)

                    if safe_word in speech_g.split():
                        self.vq.save_video(video_name="vid_micapple")
                        exit()
                        
                    self.vq.generate("/imagenet", speech_g)
                    



            except Exception as e:
                traceback.print_exc()
                print("Sorry, I did not get that")
                #exit()
