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

    def process_image(self, image_path, output_dir="responses"):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The image file '{image_path}' does not exist.")

        os.makedirs(output_dir, exist_ok=True)

        question = "Please describe this image"
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
            f.write(answer)
        
        print(f"Processed {image_path} -> {output_filename}")
        print(f"{prepare_inputs['sft_format'][0]}", answer)
        return answer

    def handle_client(self, client_socket):
        try:
            # First receive message length
            message_len_bytes = client_socket.recv(8)
            if not message_len_bytes:
                return
            message_len = int.from_bytes(message_len_bytes, byteorder='big')
            
            # Then receive the full message
            chunks = []
            bytes_received = 0
            while bytes_received < message_len:
                chunk = client_socket.recv(min(8192, message_len - bytes_received))
                if not chunk:
                    return
                chunks.append(chunk)
                bytes_received += len(chunk)
            
            data = b''.join(chunks).decode('utf-8')
            request = json.loads(data)
            
            image_paths = request.get('images', [])
            output_dir = request.get('output_dir', 'responses')
            
            results = []
            for image_path in image_paths:
                try:
                    result = self.process_image(image_path, output_dir)
                    results.append({"path": image_path, "success": True, "result": result})
                except Exception as e:
                    results.append({"path": image_path, "success": False, "error": str(e)})
            
            # Send response length first, then the response
            response = json.dumps(results).encode('utf-8')
            response_len = len(response)
            client_socket.sendall(response_len.to_bytes(8, byteorder='big'))
            client_socket.sendall(response)
        except Exception as e:
            error_response = json.dumps({"error": str(e)}).encode('utf-8')
            error_len = len(error_response)
            client_socket.sendall(error_len.to_bytes(8, byteorder='big'))
            client_socket.sendall(error_response)
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

class JanusClient:
    def __init__(self, port=65432):
        self.port = port

    def process_images(self, image_paths, output_dir="responses"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(('localhost', self.port))
            request = {
                'images': image_paths,
                'output_dir': output_dir
            }
            sock.sendall(json.dumps(request).encode('utf-8'))
            
            response = sock.recv(1024*1024).decode('utf-8')  # Increased buffer size
            return json.loads(response)
        finally:
            sock.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process images with Janus-Pro model')
    parser.add_argument('--images', help='Comma-separated list of image paths')
    parser.add_argument('image_path', nargs='?', help='Path to single image file')
    parser.add_argument('--persist', action='store_true', help='Run as persistent server')
    parser.add_argument('--output-dir', default='responses', help='Directory to save responses')
    parser.add_argument('--port', type=int, default=65432, help='Port for server mode')
    args = parser.parse_args()

    if args.persist:
        # Run as server
        server = JanusServer()
        server.run_server(port=args.port)
    else:
        # Run as client
        client = JanusClient(port=args.port)
        image_paths = []
        
        if args.images:
            image_paths.extend([path.strip() for path in args.images.split(',')])
        if args.image_path:
            image_paths.append(args.image_path)
            
        if not image_paths:
            parser.print_help()
            return

        try:
            results = client.process_images(image_paths, args.output_dir)
            for result in results:
                if result.get('success'):
                    print(f"Successfully processed {result['path']}")
                else:
                    print(f"Error processing {result['path']}: {result.get('error')}")
        except ConnectionRefusedError:
            print("Error: Could not connect to Janus server. Please start it with --persist flag first.")

if __name__ == "__main__":
    main()
