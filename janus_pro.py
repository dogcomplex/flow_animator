import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# Add Janus directory to Python path
JANUS_DIR = Path(__file__).parent / "Janus"
sys.path.append(str(JANUS_DIR))

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import socket
import json
import threading
import queue
import signal
from dotenv import load_dotenv
import argparse

class JanusServer:
    def __init__(self):
        self.cache_dir = Path("./cached_model").absolute()
        self.model_name = "deepseek-ai/Janus-Pro-7B"
        self._initialize_model()
        self.request_queue = queue.Queue()
        self.running = True

    def _initialize_model(self):
        # Ensure the cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        print("Loading VLChatProcessor...")
        self.vl_chat_processor = VLChatProcessor.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.tokenizer = self.vl_chat_processor.tokenizer

        print("Loading model...")
        self.vl_gpt = AutoModelForCausalLM.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()
        print("Model loaded and ready")

    def process_image(self, image_path, prompt=None, output_dir="responses"):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The image file '{image_path}' does not exist.")

        os.makedirs(output_dir, exist_ok=True)

        # Use provided prompt or default
        question = prompt if prompt else "Please describe this image"
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        output_filename = os.path.join(output_dir, os.path.basename(image_path).rsplit('.', 1)[0] + '.txt')
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"Prompt: {question}\nResponse: {answer}")
        
        return answer

    def handle_client(self, client_socket):
        try:
            # Receive message length
            message_len_bytes = client_socket.recv(8)
            if not message_len_bytes:
                raise Exception("No data received")
            message_len = int.from_bytes(message_len_bytes, byteorder='big')
            
            # Receive full message
            data = b''
            while len(data) < message_len:
                chunk = client_socket.recv(min(8192, message_len - len(data)))
                if not chunk:
                    raise Exception("Connection closed while receiving request")
                data += chunk
            
            request = json.loads(data.decode('utf-8'))
            
            # Handle both single image and image list formats
            image_paths = request.get('images', [])
            if not image_paths and request.get('image'):
                image_paths = [request.get('image')]
            
            output_dir = request.get('output_dir', 'responses')
            prompt = request.get('prompt', "Describe the scene content and any motion or action occurring in this frame.")
            
            results = []
            for image_path in image_paths:
                try:
                    result = self.process_image(image_path, prompt, output_dir)
                    results.append({
                        "path": image_path,
                        "success": True,
                        "result": result
                    })
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    results.append({
                        "path": image_path,
                        "success": False,
                        "error": str(e)
                    })
            
            # Send response
            response_data = json.dumps(results).encode('utf-8')
            response_len = len(response_data)
            client_socket.sendall(response_len.to_bytes(8, byteorder='big'))
            client_socket.sendall(response_data)
            
        except Exception as e:
            print(f"Server error: {str(e)}")
            error_response = json.dumps([{
                "success": False,
                "error": str(e)
            }]).encode('utf-8')
            try:
                client_socket.sendall(len(error_response).to_bytes(8, byteorder='big'))
                client_socket.sendall(error_response)
            except:
                pass
        finally:
            client_socket.close()

    def run_server(self, port=65432):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', port))
        server_socket.listen()
        print(f"Server listening on port {port}")

        def signal_handler(signum, frame):
            print("\nShutting down server...")
            self.running = False
            server_socket.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while self.running:
            try:
                client_socket, addr = server_socket.accept()
                print(f"Connected by {addr}")
                thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                thread.start()
            except Exception as e:
                if self.running:
                    print(f"Error accepting connection: {e}")

    def process_images(self, image_paths, prompts=None, output_dir="responses"):
        """
        Process multiple images with optional prompts
        Args:
            image_paths: List of image paths
            prompts: Optional list of prompts (must match length of image_paths if provided)
            output_dir: Directory to save responses
        """
        if not image_paths:
            return []
        
        if prompts and len(prompts) != len(image_paths):
            raise ValueError("Number of prompts must match number of images")
        
        results = []
        for idx, image_path in enumerate(image_paths):
            try:
                prompt = prompts[idx] if prompts else None
                result = self.process_image(image_path, prompt, output_dir)
                results.append({
                    "path": image_path,
                    "prompt": prompt,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "path": image_path,
                    "prompt": prompt if prompts else None,
                    "success": False,
                    "error": str(e)
                })
        return results

class JanusClient:
    def __init__(self, port=65432):
        self.port = port

    def process_image(self, image_path, prompt=None, output_dir="responses"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(('localhost', self.port))
            request = {
                'image': image_path,
                'prompt': prompt,
                'output_dir': output_dir
            }
            request_data = json.dumps(request).encode('utf-8')
            sock.sendall(len(request_data).to_bytes(8, byteorder='big'))
            sock.sendall(request_data)
            
            response_len = int.from_bytes(sock.recv(8), byteorder='big')
            response = b''
            while len(response) < response_len:
                chunk = sock.recv(min(8192, response_len - len(response)))
                if not chunk:
                    break
                response += chunk
            
            return json.loads(response.decode('utf-8'))
        finally:
            sock.close()

    def process_images(self, image_paths, prompts=None, output_dir="responses"):
        """
        Process multiple images with optional prompts
        Args:
            image_paths: List of image paths
            prompts: Optional list of prompts (must match length of image_paths if provided)
            output_dir: Directory to save responses
        """
        if not image_paths:
            return []
        
        if prompts and len(prompts) != len(image_paths):
            raise ValueError("Number of prompts must match number of images")
        
        results = []
        for idx, image_path in enumerate(image_paths):
            try:
                prompt = prompts[idx] if prompts else None
                result = self.process_image(image_path, prompt, output_dir)
                results.append({
                    "path": image_path,
                    "prompt": prompt,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "path": image_path,
                    "prompt": prompt if prompts else None,
                    "success": False,
                    "error": str(e)
                })
        return results

def main():
    parser = argparse.ArgumentParser(description='Process images with Janus-Pro model')
    parser.add_argument('--images', help='Comma-separated list of image paths')
    parser.add_argument('--prompts', help='Comma-separated list of prompts (must match number of images)')
    parser.add_argument('--persist', action='store_true', help='Run as persistent server')
    parser.add_argument('--output-dir', default='responses', help='Directory to save responses')
    parser.add_argument('--port', type=int, default=65432, help='Port for server mode')
    
    # Support for positional arguments
    parser.add_argument('args', nargs='*', help='Positional args: [image_paths] [prompts]')
    
    args = parser.parse_args()

    if args.persist:
        server = JanusServer()
        server.run_server(port=args.port)
        return

    # Handle image paths
    image_paths = []
    if args.images:
        image_paths.extend([path.strip() for path in args.images.split(',')])
    elif len(args.args) >= 1:
        image_paths.extend([path.strip() for path in args.args[0].split(',')])

    # Handle prompts
    prompts = None
    if args.prompts:
        prompts = [prompt.strip() for prompt in args.prompts.split(',')]
    elif len(args.args) >= 2:
        prompts = [prompt.strip() for prompt in args.args[1].split(',')]

    if not image_paths:
        parser.print_help()
        return

    try:
        client = JanusClient(port=args.port)
        results = client.process_images(image_paths, prompts, args.output_dir)
        
        for result in results:
            if result['success']:
                print(f"Successfully processed {result['path']}")
                print(f"Prompt: {result['prompt']}")
                print(f"Response: {result['result']}\n")
            else:
                print(f"Error processing {result['path']}: {result['error']}\n")
    except ConnectionRefusedError:
        print("Error: Could not connect to Janus server. Please start it with --persist flag first.")

if __name__ == "__main__":
    main()
