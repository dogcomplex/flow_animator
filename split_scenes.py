import subprocess
import os
import csv
from math import ceil
import math

# Configuration variables
INPUT_VIDEO = "input.mp4"
OUTPUT_DIR = "outputs/scenes/"
OUTPUT_CSV = "outputs/input-Scenes.csv"  # New CSV location
FRAMES_PER_SEGMENT = 70  # roughly 2 seconds at ~30fps
SCENE_FILENAME_TEMPLATE = "input-Scene-{scene_num:03d}.mp4"
SEGMENT_FILENAME_TEMPLATE = "input-Scene-{scene_num:03d}-{segment:03d}.mp4"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)  # Ensure CSV directory exists

# Step 1: Detect scenes and save the scene list to a CSV
print("Detecting scenes...")
subprocess.run([
    "scenedetect", "-i", INPUT_VIDEO, "detect-content",
    "list-scenes", "-o", os.path.dirname(OUTPUT_CSV)  # Change output directory for CSV
])

# Step 2: Generate split video files for each detected scene
print("Splitting scenes into separate video files...")
subprocess.run([
    "scenedetect", "-i", INPUT_VIDEO, "detect-content",
    "split-video", "-o", OUTPUT_DIR, 
    "--filename", "input-Scene-$SCENE_NUMBER"
])

# Step 3: Load the scene CSV and process further splits for scenes longer than FRAMES_PER_SEGMENT
print(f"\nLooking for CSV file: {OUTPUT_CSV}")

if os.path.exists(OUTPUT_CSV):
    print(f"Found CSV file. Processing scenes...")
    with open(OUTPUT_CSV, 'r') as f:
        csv_content = f.readlines()
        
    print(f"CSV lines read: {len(csv_content)}")
    
    # Skip header and separator
    scene_lines = [line.strip() for line in csv_content[2:] if '----' not in line]
    print(f"Scene lines found: {len(scene_lines)}")
    
    for line in scene_lines:
        # New CSV format has 10 columns:
        # scene_num, start_frame, start_time, start_time_sec, end_frame, end_time, end_time_sec, duration_frames, duration_time, duration_sec
        parts = line.split(',')
        
        if len(parts) == 10:  # Verify we have all expected columns
            scene_num = int(parts[0])
            start_frame = int(parts[1])
            end_frame = int(parts[4])
            duration = end_frame - start_frame + 1
            
            input_file = os.path.join(OUTPUT_DIR, SCENE_FILENAME_TEMPLATE.format(scene_num=scene_num))
            
            if duration > FRAMES_PER_SEGMENT:
                print(f"\nProcessing scene {scene_num:03d} ({duration} frames)")
                
                if os.path.exists(input_file):
                    # Process longer scenes into segments
                    num_segments = math.ceil(duration / FRAMES_PER_SEGMENT)
                    segments_created = []
                    
                    for segment in range(num_segments):
                        segment_start = segment * FRAMES_PER_SEGMENT / 24  # Convert frames to seconds
                        segment_duration = min(FRAMES_PER_SEGMENT/24, (duration - segment * FRAMES_PER_SEGMENT)/24)
                        output_file = os.path.join(OUTPUT_DIR, SEGMENT_FILENAME_TEMPLATE.format(scene_num=scene_num, segment=segment+1))
                        segments_created.append(output_file)
                        
                        print(f"    Segment {segment+1}/{num_segments}:")
                        print(f"      Start: {segment_start:.2f}s")
                        print(f"      Duration: {segment_duration:.2f}s")
                        print(f"      Output: {output_file}")
                        
                        subprocess.run([
                            "ffmpeg", "-i", input_file,
                            "-ss", str(segment_start),
                            "-t", str(segment_duration),
                            "-c:v", "libx264", "-preset", "veryfast",
                            output_file
                        ])
                    
                    # After all segments are created successfully, delete the parent clip
                    if all(os.path.exists(segment) for segment in segments_created):
                        print(f"  Deleting parent clip: {input_file}")
                        os.remove(input_file)
                    else:
                        print(f"  Warning: Not all segments were created, keeping parent clip")
                else:
                    print(f"  Source file not found: {input_file}")
        else:
            print(f"Warning: Unexpected CSV format - expected 10 columns, got {len(parts)}")
else:
    print(f"ERROR: Scene list CSV not found!")
