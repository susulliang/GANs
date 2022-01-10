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
        prompts = "winter day with some trees / a snowy russian forest / forest with very beautiful landscape / winter day with some trees / a snowy russian forest / forest with very beautiful landscape"
        prompts = prompts.split(" / ")
        for input_prompt in prompts:
            self.vq.generate("/imagenet", input_prompt)
        

        self.vq.save_video(video_name="vid_interp_out")

    def test_image_prompts(self, max_iter=120):
        target_image_file = "tdout_noise.jpg"
        print(f" [DEBUG] Using {target_image_file} as target image")

        for i in range(max_iter):
            self.vq.generate(
                channel="/imagenet",
                target_image=target_image_file,
                ramdisk=True)
                

        self.vq.save_video(video_name="vid_interp_out")

