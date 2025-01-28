import os
import shutil
import argparse
from pathlib import Path
from scene_analyzer import SceneAnalyzer, JanusAnalyzer
from typing import Dict, List
from tqdm import tqdm

# Configuration
OUTPUT_BASE = "outputs"
OUTPUT_DIR = f"{OUTPUT_BASE}/scenes"
PROMPTS_DIR = f"{OUTPUT_BASE}/scene_prompts"
FRAMES_DIR = f"{OUTPUT_BASE}/frames"
FRAME_PROMPTS_DIR = f"{OUTPUT_BASE}/frame_prompts"
API_ENDPOINT = "http://192.168.56.1:12345"
FRAME_SKIP = 24  # Analyze every 24th frame
JANUS_PATH = "./Janus/janus_pro.py"
JANUS_MODE = True

def reset_cache():
    """Clear all cache directories"""
    Path(OUTPUT_BASE).mkdir(exist_ok=True)
    dirs_to_clear = [
        f'{OUTPUT_BASE}/frames', 
        f'{OUTPUT_BASE}/scene_prompts', 
        f'{OUTPUT_BASE}/frame_prompts'
    ]
    for dir_name in dirs_to_clear:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"Cleared {dir_name} directory")
        Path(dir_name).mkdir(exist_ok=True)

def save_prompt_for_file(prompt: str, filename: str):
    """Save the same prompt for both main clips and their segments"""
    prompt_path = Path(PROMPTS_DIR) / f"{filename}.txt"
    prompt_path.write_text(prompt)

def main():
    parser = argparse.ArgumentParser(description='Process video scenes and generate descriptions')
    parser.add_argument('--reset', action='store_true', help='Clear all cache directories before processing')
    args = parser.parse_args()

    if args.reset:
        reset_cache()
    
    Path(PROMPTS_DIR).mkdir(exist_ok=True)
    
    # Initialize appropriate analyzer based on mode
    analyzer = JanusAnalyzer(frame_skip=FRAME_SKIP, prompts_dir=PROMPTS_DIR) if JANUS_MODE else \
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
                description = analyzer.analyze_frames(frames, str(video_path))
                scene_descriptions[scene_num] = description
                analyzer.save_prompt_cache(scene_num, description)
                for video_path in scene_groups[scene_num]:
                    save_prompt_for_file(description, video_path.stem)
                print(f"\nScene {scene_num:03d}: {description}")

if __name__ == "__main__":
    main()
