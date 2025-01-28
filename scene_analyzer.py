import os
import cv2
import requests
import json
import base64
import numpy
from pathlib import Path
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import subprocess

class SceneAnalyzer:
    def __init__(self, api_endpoint: str, frame_skip: int = 5, prompts_dir: str = "scene_prompts"):
        self.api_endpoint = api_endpoint
        self.frame_skip = frame_skip
        self.prompts_dir = prompts_dir
        Path(prompts_dir).mkdir(exist_ok=True)
        self.cache: Dict[str, str] = {}

    def get_cache_path(self, scene_num: int) -> Path:
        return Path(self.prompts_dir) / f"scene_{scene_num:03d}.txt"

    def load_cached_prompt(self, scene_num: int) -> Optional[str]:
        cache_path = self.get_cache_path(scene_num)
        if cache_path.exists():
            return cache_path.read_text()
        return None

    def save_prompt_cache(self, scene_num: int, prompt: str):
        cache_path = self.get_cache_path(scene_num)
        cache_path.write_text(prompt)

    def extract_scene_info(self, filename: str) -> Optional[tuple[int, Optional[int]]]:
        """Extract scene and segment numbers from filename"""
        try:
            # Remove file extension
            name = filename.rsplit('.', 1)[0]
            parts = name.split('-')
            
            # Handle both formats:
            # input-Scene-001.mp4 and input-Scene-001-001.mp4
            if len(parts) >= 3 and parts[1].lower() == "scene":
                scene_num = int(parts[2])
                segment_num = int(parts[3]) if len(parts) >= 4 else None
                return (scene_num, segment_num)
            return None
        except (ValueError, IndexError):
            return None

    def get_frames(self, video_path: str) -> List[numpy.ndarray]:
        """Extract frames from video with frame skipping"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.frame_skip == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1
            
        cap.release()
        return frames

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze_frames(self, frames: List[numpy.ndarray]) -> str:
        encoded_frames = [self._encode_frame(frame) for frame in frames]
        
        payload = {
            "model": "liuhaotian_llava-v1.5-13b",
            "messages": [
                {
                    "role": "user",
                    "content": "Describe the scene content and any motion or action occurring in these frames.",
                    "images": encoded_frames
                }
            ]
        }
        
        response = requests.post(
            f"{self.api_endpoint}/v1/chat/completions", 
            json=payload, 
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}")
        
        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected API response format: {str(e)}")

    def _encode_frame(self, frame: numpy.ndarray) -> str:
        """Convert frame to base64 string"""
        success, encoded = cv2.imencode('.jpg', frame)
        return base64.b64encode(encoded.tobytes()).decode('utf-8')

class JanusAnalyzer(SceneAnalyzer):
    def __init__(self, janus_path: str, frame_skip: int = 5, prompts_dir: str = "scene_prompts"):
        super().__init__("", frame_skip, prompts_dir)  # Empty API endpoint as it's not used
        self.janus_path = janus_path
        self.temp_dir = Path("temp_frames")
        self.temp_dir.mkdir(exist_ok=True)

    def analyze_frames(self, frames: List[numpy.ndarray]) -> str:
        """Analyze frames using Janus-Pro model"""
        # Save frames as temporary images
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = self.temp_dir / f"frame_{i:03d}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_paths.append(frame_path)

        # Create comma-separated list of image paths
        images_arg = ",".join(str(p) for p in frame_paths)
        
        print("\nRunning Janus-Pro analysis...")
        # Run Janus-Pro and capture all output
        result = subprocess.run([
            "python", self.janus_path,
            frame_paths[0],  # First image as main argument
            "--images", images_arg,
            "--output-dir", self.temp_dir
        ], capture_output=True, text=True)

        # Print all stdout and stderr
        if result.stdout:
            print("\nJanus Output:")
            print(result.stdout)
        if result.stderr:
            print("\nJanus Errors/Warnings:")
            print(result.stderr)

        # Clean up temporary images
        for path in frame_paths:
            path.unlink()

        # Read the output file (matches first image name)
        output_file = self.temp_dir / f"{frame_paths[0].stem}.txt"
        if output_file.exists():
            description = output_file.read_text().strip()
            output_file.unlink()
            return description
        
        raise Exception("Janus-Pro analysis failed to produce output")

    def analyze_frames_batch(self, frames: List[numpy.ndarray]) -> List[str]:
        """Analyze multiple frames in a single batch using Janus-Pro model"""
        # Save frames as temporary images
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = self.temp_dir / f"frame_{i:03d}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_paths.append(frame_path)

        # Create comma-separated list of image paths
        images_arg = ",".join(str(p) for p in frame_paths)
        
        print("\nRunning Janus-Pro batch analysis...")
        # Run Janus-Pro with all frames at once
        result = subprocess.run([
            "python", self.janus_path,
            frame_paths[0],  # First image as main argument
            "--images", images_arg,
            "--output-dir", self.temp_dir
        ], capture_output=True, text=True)

        # Print all stdout and stderr
        if result.stdout:
            print("\nJanus Output:")
            print(result.stdout)
        if result.stderr:
            print("\nJanus Errors/Warnings:")
            print(result.stderr)

        # Read all output files and collect descriptions
        descriptions = []
        for frame_path in frame_paths:
            output_file = self.temp_dir / f"{frame_path.stem}.txt"
            if output_file.exists():
                descriptions.append(output_file.read_text().strip())
                output_file.unlink()
            else:
                descriptions.append("")  # Empty description for failed analyses
            frame_path.unlink()

        if not any(descriptions):
            raise Exception("Janus-Pro analysis failed to produce any output")
        
        return descriptions

    def aggregate_descriptions(self, descriptions: List[str]) -> str:
        """Combine multiple frame descriptions into a single scene description"""
        # For now, just use the first non-empty description
        # This could be enhanced with more sophisticated aggregation
        for desc in descriptions:
            if desc.strip():
                return desc
        return "No description available" 