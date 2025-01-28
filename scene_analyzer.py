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
import threading
import time
import socket
import sys

class SceneAnalyzer:
    def __init__(self, api_endpoint: str, frame_skip: int = 5, prompts_dir: str = "outputs/scene_prompts"):
        self.api_endpoint = api_endpoint
        self.frame_skip = frame_skip
        self.prompts_dir = prompts_dir
        self.frames_dir = "outputs/frames"
        self.frame_prompts_dir = "outputs/frame_prompts"
        Path(prompts_dir).mkdir(exist_ok=True, parents=True)
        Path(self.frames_dir).mkdir(exist_ok=True, parents=True)
        Path(self.frame_prompts_dir).mkdir(exist_ok=True, parents=True)

    def get_prompt_path(self, video_path: str) -> Path:
        """Get path for prompt cache based on video filename"""
        return Path(self.prompts_dir) / f"{Path(video_path).stem}.txt"

    def load_cached_prompt(self, scene_num: int) -> Optional[str]:
        """Load cached prompt using scene number"""
        # Convert scene number to expected video filename
        video_name = f"input-Scene-{scene_num:03d}"
        prompt_path = Path(self.prompts_dir) / f"{video_name}.txt"
        if prompt_path.exists():
            return prompt_path.read_text()
        return None

    def save_prompt_cache(self, scene_num: int, prompt: str):
        """Save prompt using scene number - now just an alias for save_prompt_for_file"""
        video_name = f"input-Scene-{scene_num:03d}"
        self.save_prompt_for_file(prompt, video_name)

    def save_prompt_for_file(self, prompt: str, filename: str):
        """Save prompt using video filename"""
        prompt_path = Path(self.prompts_dir) / f"{filename}.txt"
        prompt_path.write_text(prompt)

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

    def get_cache_paths(self, video_path: str) -> tuple[Path, Path]:
        """Get paths for frame cache and prompt cache"""
        base_name = Path(video_path).stem + "_f24"
        frame_path = Path(self.frames_dir) / f"{base_name}.npy"
        prompt_path = Path(self.frame_prompts_dir) / f"{base_name}.txt"
        return frame_path, prompt_path

    def get_frames(self, video_path: str) -> List[numpy.ndarray]:
        """Extract frames from video with adaptive frame skipping and caching"""
        base_name = Path(video_path).stem
        frames = []
        
        # Check if we have all cached frame images
        frame_count = 0
        while True:
            frame_path = Path(self.frames_dir) / f"{base_name}_f{frame_count:03d}.png"
            if not frame_path.exists():
                break
            frame = cv2.imread(str(frame_path))
            if frame is None:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1
        
        if frames:  # If we found cached frames, return them
            return frames
        
        # No cached frames found, process video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        effective_skip = min(self.frame_skip, total_frames)
        
        frame_count = 0
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % effective_skip == 0 or frame_count == total_frames - 1:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Save frame as PNG
                frame_path = Path(self.frames_dir) / f"{base_name}_f{frame_index:03d}.png"
                cv2.imwrite(str(frame_path), frame)
                frame_index += 1
            frame_count += 1
        
        cap.release()
        return frames

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze_frames(self, frames: List[numpy.ndarray], video_path: str) -> str:
        """Analyze frames with caching"""
        _, prompt_cache_path = self.get_cache_paths(video_path)
        
        # Create frame_prompts directory if it doesn't exist
        prompt_cache_path.parent.mkdir(exist_ok=True)
        
        # Check for cached prompt
        if prompt_cache_path.exists():
            return prompt_cache_path.read_text()
        
        # Original analysis code
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
            description = response.json()["choices"][0]["message"]["content"]
            # Cache the prompt
            prompt_cache_path.write_text(description)
            return description
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected API response format: {str(e)}")

    def _encode_frame(self, frame: numpy.ndarray) -> str:
        """Convert frame to base64 string"""
        success, encoded = cv2.imencode('.jpg', frame)
        return base64.b64encode(encoded.tobytes()).decode('utf-8')

class JanusAnalyzer(SceneAnalyzer):
    def __init__(self, frame_skip: int = 5, prompts_dir: str = "outputs/scene_prompts"):
        super().__init__("", frame_skip, prompts_dir)
        self.janus_path = "janus_pro.py"
        Path("outputs/frames").mkdir(exist_ok=True)
        Path("outputs/frame_prompts").mkdir(exist_ok=True)
        self.server_process = None
        self.start_server()
        import atexit
        atexit.register(self.cleanup)

    def start_server(self):
        print("Starting Janus server...")
        try:
            # Try to connect first to see if server is already running
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', 65432))
            sock.close()
            print("Janus server already running")
            return
        except ConnectionRefusedError:
            pass

        # Start the server process
        server_cmd = [sys.executable, "janus_pro.py", "--persist"]
        self.server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start (look for "Server listening" message)
        max_retries = 30
        retry_count = 0
        while retry_count < max_retries:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(('localhost', 65432))
                sock.close()
                print("Janus server started successfully")
                return
            except ConnectionRefusedError:
                time.sleep(1)
                retry_count += 1
        
        # If we get here, server failed to start
        if self.server_process:
            stdout, stderr = self.server_process.communicate()
            print("Server stdout:", stdout)
            print("Server stderr:", stderr)
            self.server_process.terminate()
        raise Exception("Failed to start Janus server after 30 seconds")

    def cleanup(self):
        """Cleanup only server resources, keeping frame caches"""
        if self.server_process:
            print("\nShutting down Janus server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None

    def analyze_frames(self, frames: List[numpy.ndarray], video_path: str) -> str:
        """Analyze frames using Janus-Pro model with persistent caching"""
        base_name = Path(video_path).stem
        descriptions = []
        total_frames = len(frames)
        
        print(f"\nProcessing {total_frames} frames...")
        
        for i, frame in enumerate(frames):
            frame_path = Path(self.frames_dir) / f"{base_name}_f{i:03d}.png"
            prompt_path = Path(self.frame_prompts_dir) / f"{base_name}_f{i:03d}.txt"
            
            # Check if we already have a cached prompt
            if prompt_path.exists():
                with open(prompt_path, 'r') as f:
                    description = f.read().strip()
                    if description:  # Only append non-empty descriptions
                        descriptions.append(description)
                print(f"\nFrame {i+1}/{total_frames}: Using cached description")
                continue
            
            # Save frame as image
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(30)  # Add timeout
                    sock.connect(('localhost', 65432))
                    
                    request = {
                        'images': [str(frame_path)],
                        'output_dir': str(Path(self.frame_prompts_dir))
                    }
                    
                    message = json.dumps(request).encode('utf-8')
                    message_len = len(message)
                    sock.sendall(message_len.to_bytes(8, byteorder='big'))
                    sock.sendall(message)
                    
                    response_len_bytes = sock.recv(8)
                    if not response_len_bytes:
                        raise Exception("Connection closed by server")
                        
                    response_len = int.from_bytes(response_len_bytes, byteorder='big')
                    response = b''
                    while len(response) < response_len:
                        chunk = sock.recv(min(8192, response_len - len(response)))
                        if not chunk:
                            raise Exception("Connection closed while receiving data")
                        response += chunk
                    
                    results = json.loads(response.decode('utf-8'))
                    if results and isinstance(results[0], dict) and results[0].get('success'):
                        description = results[0]['result']
                        if description.strip():  # Only append non-empty descriptions
                            descriptions.append(description)
                        prompt_path.write_text(description)
                        print(f"\nFrame {i+1}/{total_frames}: {description}")
                        break  # Success - exit retry loop
                    else:
                        error = results[0].get('error') if results else "Unknown error"
                        raise Exception(error)
                        
                except Exception as e:
                    retry_count += 1
                    print(f"\nError processing frame {i+1}/{total_frames} (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count == max_retries:
                        print(f"Failed to process frame after {max_retries} attempts")
                finally:
                    sock.close()
                
        # Only raise exception if we got no descriptions at all
        if not descriptions:
            raise Exception("Janus-Pro analysis failed to produce any output")
        
        # Return the first good description
        return descriptions[0] 