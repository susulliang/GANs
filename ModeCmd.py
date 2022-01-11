
import speech_recognition as sr
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

    def use_mic_inputs(self):
        # Initialize recognizer class (for recognizing the speech)
        r = sr.Recognizer()

        # Reading Microphone as source
        # listening the speech and store in audio_text variable

        
        print("Waiting for mic prompts...")
        while True:
            try:
                with sr.Microphone() as source:
                    # -> listen for prompts
                    r.non_speaking_duration = 0.5
                    r.pause_threshold = 0.6
                    audio_text = r.listen(source, phrase_time_limit=5, timeout=10)
                    speech = r.recognize_sphinx(audio_text)
                    speech_g = r.recognize_google(audio_text)

                    print(speech)
                    print(speech_g)

                    self.vq.generate("/imagenet", speech_g)
                    

                    if "apple" in speech_g.split():
                        print("don't say apple")
                        self.vq.save_video(video_name="vid_micapple")
                        exit()

            except Exception as e:
                traceback.print_exc()
                print("Sorry, I did not get that")
                #exit()
