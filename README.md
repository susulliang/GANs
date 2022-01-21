# GANs
realtime VQGAN + CLIP + etc.
work in progress.

# Exhibition Installation
## Interactive modes
- commmand prompt
- webcam input (e.g. CAMO Webcam using iPhone 6s)
- microphone speech input

## Standby modes
- local text prompts
- local icon burn-ins
- local image prompts

## How to interact
```
python run.py
```
once the exhibition mode is running, it will enter loop synced at each second.
Type anything in the console to enter user commmand mode

Some examples of what you can prompt the AI to paint:
```
A boy holding an apple in abstract style
The girl with a pearl earring
Luncheon of the boating party
Mars and Venus allegory of peace in black and white
Adoration of the Magi
The Last Supper in simplistic lines
Watson and the shark
A banquet with the skeleton in black and white
Composition VIII
Two lovers (The lovers)
Big lemons on a tree with pink sky and blue clouds
The Anatomy Lesson of Dr. Nicolaes Tulp
View of Delft Townhall from the North 
Cubist Animal Artworks
Picasso paints his dreams
Greek Mythology
The Rialto Bridge and the Grand Canal in Venice
View of Delft
The steppe
```

Or type even in some styles to guide AI to a stylistic path. Examples (but not limited to):
```
styles = ["fish eye view", "dreaming", "Kandinsy", "Cubist", "Abstract",
             "cartoon", "watercolor", "pixel", "pixelated", "simple", "pointillism", "acrylic", 
             "Ghost in the Shell style", "Death Stranding style", "gouache", "vignetting",
             "Dark and clean background", "small and center", "geometric", "linear", "Junji Ito style"]
```
