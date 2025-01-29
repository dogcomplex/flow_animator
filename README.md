# Flow Animator

(Aka Brodyquest Animator)

Takes an input video and automatically splits it into scenes, then describes each scene using vision LLM Janus-Pro (20GB)

Meant to be paired with [kijai's CogVideoXWrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper/tree/main) implementing [Go-With-The-Flow](https://eyeline-research.github.io/Go-with-the-Flow/)

Crude workflow: [V2V_CogXFlow](V2V_CogXFlow.json)

I recommend Pinokio "Comfy Environment Manager" for ComfyUI installation, or StabilityMatrix if you're new to ComfyUI.


# INSTALLATION
```
pip install -r requirements.txt

# optional, but recommended for quality vision LLM:
git clone https://github.com/deepseek-ai/Janus

# put your input.mp4 in the same directory as split_scenes.py
# adjust configs in each file (ALLCAPS) to taste.  Recommended 70 frames maxish for 24GB vram on cogvideoX
```

# USAGE

```
python split_scenes.py
python describe_scenes.py
```

