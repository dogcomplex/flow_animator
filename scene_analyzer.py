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
        super().__init__("", frame_skip, prompts_dir)
        self.janus_path = janus_path
        self.temp_dir = Path("temp_frames")
        self.temp_dir.mkdir(exist_ok=True)
        self.server_process = None
        self.start_server()
        import atexit
        atexit.register(self.cleanup)

    def start_server(self):
        """Start the Janus server if it's not already running"""
        # Check if server is already running
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(('localhost', 65432))
            sock.close()
            print("Janus server already running")
            return
        except ConnectionRefusedError:
            pass

        print("Starting Janus server...")
        self.server_process = subprocess.Popen(
            ["python", self.janus_path, "--persist"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Wait for server to start
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(('localhost', 65432))
                sock.close()
                print("Janus server started successfully")
                
                # Start output monitoring thread
                def monitor_output():
                    while self.server_process:
                        output = self.server_process.stdout.readline()
                        if output:
                            print("Janus Server:", output.strip())
                        if not output and self.server_process.poll() is not None:
                            break
                
                threading.Thread(target=monitor_output, daemon=True).start()
                return
            except ConnectionRefusedError:
                time.sleep(1)
                if attempt == max_attempts - 1:
                    raise Exception("Failed to start Janus server")

    def cleanup(self):
        """Cleanup resources including shutting down the server"""
        if self.server_process:
            print("\nShutting down Janus server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
        
        # Cleanup temp directory
        if self.temp_dir.exists():
            for file in self.temp_dir.glob("*"):
                try:
                    file.unlink()
                except:
                    pass
            try:
                self.temp_dir.rmdir()
            except:
                pass

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
        descriptions = []
        total_frames = len(frames)
        
        print(f"\nProcessing {total_frames} frames...")
        
        for i, frame in enumerate(frames):
            # Save frame as temporary image
            frame_path = self.temp_dir / f"frame_{i:03d}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Create request for server
            request = {
                'images': [str(frame_path)],
                'output_dir': str(self.temp_dir)
            }
            
            # Connect to server and send request
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect(('localhost', 65432))
                
                # Send message length first, then the message
                message = json.dumps(request).encode('utf-8')
                message_len = len(message)
                sock.sendall(message_len.to_bytes(8, byteorder='big'))
                sock.sendall(message)
                
                # First receive the response length
                response_len_bytes = sock.recv(8)
                if not response_len_bytes:
                    raise Exception("Connection closed by server")
                response_len = int.from_bytes(response_len_bytes, byteorder='big')
                
                # Then receive the full response
                chunks = []
                bytes_received = 0
                while bytes_received < response_len:
                    chunk = sock.recv(min(8192, response_len - bytes_received))
                    if not chunk:
                        raise Exception("Connection closed by server")
                    chunks.append(chunk)
                    bytes_received += len(chunk)
                
                response = b''.join(chunks).decode('utf-8')
                
                try:
                    results = json.loads(response)
                    if results and isinstance(results[0], dict) and results[0].get('success'):
                        descriptions.append(results[0]['result'])
                        print(f"\nFrame {i+1}/{total_frames}: {results[0]['result']}")
                    else:
                        error = results[0].get('error') if results and isinstance(results[0], dict) else 'Invalid response'
                        print(f"\nError processing frame {i+1}/{total_frames}: {error}")
                        descriptions.append("")
                except json.JSONDecodeError as e:
                    print(f"\nError decoding server response for frame {i+1}/{total_frames}: {e}")
                    descriptions.append("")
                    
            except Exception as e:
                print(f"\nError communicating with Janus server for frame {i+1}/{total_frames}: {e}")
                descriptions.append("")
            finally:
                sock.close()
                try:
                    frame_path.unlink()
                except FileNotFoundError:
                    pass

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