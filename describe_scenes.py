import os
from pathlib import Path
from scene_analyzer import SceneAnalyzer, JanusAnalyzer
from typing import Dict, List
from tqdm import tqdm

# Configuration
OUTPUT_DIR = "output"
PROMPTS_DIR = "scene_prompts"
API_ENDPOINT = "http://192.168.56.1:12345"
FRAME_SKIP = 5  # Analyze every 5th frame
JANUS_PATH = "./Janus/janus_pro.py"
JANUS_MODE = True
BATCH_SIZE = 1000  # Max number of files to pass to Janus at a time

def save_prompt_for_file(prompt: str, filename: str):
    """Save the same prompt for both main clips and their segments"""
    prompt_path = Path(PROMPTS_DIR) / f"{filename}.txt"
    prompt_path.write_text(prompt)

def main():
    Path(PROMPTS_DIR).mkdir(exist_ok=True)
    
    # Initialize appropriate analyzer based on mode
    analyzer = JanusAnalyzer(JANUS_PATH, FRAME_SKIP, PROMPTS_DIR) if JANUS_MODE else \
              SceneAnalyzer(API_ENDPOINT, FRAME_SKIP, PROMPTS_DIR)
              
    scene_descriptions: Dict[int, str] = {}
    
    # Get all MP4 files in output directory
    video_files = sorted(Path(OUTPUT_DIR).glob("*.mp4"))
    print(f"Found {len(video_files)} video files")
    
    # Group files by scene number
    scene_groups: Dict[int, List[Path]] = {}
    uncached_scenes = []
    
    for video_path in video_files:
        scene_info = analyzer.extract_scene_info(video_path.name)
        if scene_info:
            scene_num, _ = scene_info
            if scene_num not in scene_groups:
                scene_groups[scene_num] = []
                # Check if scene is cached
                if not analyzer.load_cached_prompt(scene_num):
                    uncached_scenes.append(scene_num)
            scene_groups[scene_num].append(video_path)

    # Process cached scenes first
    for scene_num in sorted(scene_groups.keys()):
        if scene_num not in uncached_scenes:
            cached_prompt = analyzer.load_cached_prompt(scene_num)
            print(f"\nScene {scene_num:03d}: Using cached prompt")
            scene_descriptions[scene_num] = cached_prompt
            for video_path in scene_groups[scene_num]:
                save_prompt_for_file(cached_prompt, video_path.stem)

    # Process uncached scenes
    if uncached_scenes:
        print(f"\nProcessing {len(uncached_scenes)} uncached scenes...")
        
        # Process each scene individually
        for scene_num in uncached_scenes:
            video_path = scene_groups[scene_num][0]
            frames = analyzer.get_frames(str(video_path))
            if frames:
                description = analyzer.analyze_frames(frames)
                scene_descriptions[scene_num] = description
                analyzer.save_prompt_cache(scene_num, description)
                for video_path in scene_groups[scene_num]:
                    save_prompt_for_file(description, video_path.stem)
                print(f"\nScene {scene_num:03d}: {description}")

if __name__ == "__main__":
    main()
