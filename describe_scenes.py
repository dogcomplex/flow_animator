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

    # Process uncached scenes in batches
    if uncached_scenes:
        print(f"\nProcessing {len(uncached_scenes)} uncached scenes...")
        all_frames = []
        frame_map = []  # Keep track of which frames belong to which scene
        
        # Extract frames from all uncached scenes
        for scene_num in uncached_scenes:
            video_path = scene_groups[scene_num][0]
            frames = analyzer.get_frames(str(video_path))
            if frames:
                all_frames.extend(frames)
                frame_map.extend([scene_num] * len(frames))

        if all_frames:
            # Process all frames in one batch
            descriptions = analyzer.analyze_frames_batch(all_frames)
            
            # Map descriptions back to scenes
            current_scene = None
            current_desc = []
            
            for frame_num, (scene_num, desc) in enumerate(zip(frame_map, descriptions)):
                if current_scene != scene_num:
                    if current_scene is not None:
                        # Save the aggregated description for the previous scene
                        final_desc = analyzer.aggregate_descriptions(current_desc)
                        scene_descriptions[current_scene] = final_desc
                        analyzer.save_prompt_cache(current_scene, final_desc)
                        for video_path in scene_groups[current_scene]:
                            save_prompt_for_file(final_desc, video_path.stem)
                        print(f"\nScene {current_scene:03d}: {final_desc}")
                    
                    current_scene = scene_num
                    current_desc = [desc]
                else:
                    current_desc.append(desc)
            
            # Handle the last scene
            if current_scene is not None:
                final_desc = analyzer.aggregate_descriptions(current_desc)
                scene_descriptions[current_scene] = final_desc
                analyzer.save_prompt_cache(current_scene, final_desc)
                for video_path in scene_groups[current_scene]:
                    save_prompt_for_file(final_desc, video_path.stem)
                print(f"\nScene {current_scene:03d}: {final_desc}")

if __name__ == "__main__":
    main()
