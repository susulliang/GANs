import speech_recognition as sr
from googletrans import Translator, constants
import traceback

class CmdHandle:
    # -> Command Line Handler 
    def __init__(self, vq_ref) -> None:
        self.vq = vq_ref
        pass

    def start_looping(self):
        while True:
            channel, input_prompt = self.process_input(
                input("Your prompt here: "))
            self.vq.generate(channel, input_prompt)

    def process_input(input_prompt):
        input_prompt = input_prompt.strip()
        return "/wikiart", input_prompt

    def test_prompts(self):
        print(" [DEBUG] Using default test prompt sequence")

        # Read prompts from local samples
        txt_file = open('prompts.txt', 'r')
        lines = txt_file.readlines()

        # Strips the newline character
        for line in lines[:3]:
            parse = line.strip().split('.')
            self.vq.generate("/imagenet", parse[1])


        self.vq.save_video(video_name="vid_interp_out")

    def test_cmd_inputs(self):
        print(" [DEBUG] Using default test prompt sequence")
        prompts = "cube / spiral / ocean and beach / night and moon / dark sky and moon / green and orange colors / broccoli and vegetable"
        prompts = "wassily kandinsky style landscapes / wassily kandinsky style cubes / beautiful sky"
        prompts = prompts.split(" / ")
        for input_prompt in prompts:
            self.vq.generate("/imagenet", input_prompt)
        

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
