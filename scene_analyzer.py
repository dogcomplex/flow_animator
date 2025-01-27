import os
import cv2
import requests
import json
import base64
import numpy
from pathlib import Path
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

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