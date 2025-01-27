import os
from pathlib import Path
from scene_analyzer import SceneAnalyzer
from typing import Dict, List
from tqdm import tqdm

# Configuration
OUTPUT_DIR = "output"
PROMPTS_DIR = "scene_prompts"
API_ENDPOINT = "http://192.168.56.1:12345"
FRAME_SKIP = 5  # Analyze every 5th frame

def save_prompt_for_file(prompt: str, filename: str):
    """Save the same prompt for both main clips and their segments"""
    prompt_path = Path(PROMPTS_DIR) / f"{filename}.txt"
    prompt_path.write_text(prompt)

def main():
    Path(PROMPTS_DIR).mkdir(exist_ok=True)
    analyzer = SceneAnalyzer(API_ENDPOINT, FRAME_SKIP, PROMPTS_DIR)
    scene_descriptions: Dict[int, str] = {}
    
    # Get all MP4 files in output directory
    video_files = sorted(Path(OUTPUT_DIR).glob("*.mp4"))
    print(f"Found {len(video_files)} video files")
    
    # Group files by scene number
    scene_groups: Dict[int, List[Path]] = {}
    for video_path in video_files:
        scene_info = analyzer.extract_scene_info(video_path.name)
        if scene_info:
            scene_num, _ = scene_info
            if scene_num not in scene_groups:
                scene_groups[scene_num] = []
            scene_groups[scene_num].append(video_path)
    
    # Process each scene
    for scene_num in tqdm(sorted(scene_groups.keys()), desc="Processing scenes"):
        # Check cache first
        cached_prompt = analyzer.load_cached_prompt(scene_num)
        if cached_prompt:
            print(f"\nScene {scene_num:03d}: Using cached prompt")
            scene_descriptions[scene_num] = cached_prompt
            # Save prompt for all segment files
            for video_path in scene_groups[scene_num]:
                save_prompt_for_file(cached_prompt, video_path.stem)
            continue
            
        try:
            # Use the first file in the group for analysis
            video_path = scene_groups[scene_num][0]
            frames = analyzer.get_frames(str(video_path))
            
            if not frames:
                print(f"\nWarning: No frames extracted from {video_path}")
                continue
                
            description = analyzer.analyze_frames([frames[0]])
            scene_descriptions[scene_num] = description
            
            # Cache the result
            analyzer.save_prompt_cache(scene_num, description)
            
            # Save prompt for all files in this scene group
            for video_path in scene_groups[scene_num]:
                save_prompt_for_file(description, video_path.stem)
                
            print(f"\nScene {scene_num:03d}: {description}")
            
        except Exception as e:
            print(f"\nError processing scene {scene_num:03d}: {str(e)}")

if __name__ == "__main__":
    main()
