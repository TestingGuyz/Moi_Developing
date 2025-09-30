import os
import json
import base64
import cv2
import numpy as np
import requests
import re
import html
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from groq import Groq
from tavily import TavilyClient
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import torch
from PIL import Image
import io
from dotenv import load_dotenv
import threading
import time

# Import new modules
from knowledge_database import KnowledgeDatabase
from agentic_ai import AgenticSystem
from image_generation import ImageGenerator

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

class AIAssistant:
    def __init__(self):
        # Initialize API clients
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
        
        # Initialize Knowledge Database
        self.knowledge_db = KnowledgeDatabase()
        
        # Initialize Image Generator
        self.image_generator = ImageGenerator()
        
        # Initialize Agentic AI System
        self.agentic_system = AgenticSystem(
            knowledge_db=self.knowledge_db,
            tavily_api_key=os.getenv('TAVILY_API_KEY'),
            groq_api_key=os.getenv('GROQ_API_KEY')
        )
        
        # Start all autonomous agents
        try:
            self.agentic_system.start_all_agents()
            print("✅ Agentic AI systems started successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not start all agentic systems: {e}")
        
        # Initialize vision models
        self.setup_vision_models()
        
        # Initialize object detection
        self.setup_object_detection()
        
        # Camera variables
        self.camera = None
        self.vision_mode = False
        self.current_frame = None  # Store current frame for analysis
        self.frame_lock = threading.Lock()
        self.chat_temperature = float(os.getenv('DEFAULT_TEMPERATURE', '0.7'))
        
        # Chat history (will be managed by frontend localStorage)
        self.system_prompt = """You are Moi, an advanced AI assistant with vision, memory, and learning capabilities. 
        You can see objects, describe them, remember information, and have conversations. 
        Be helpful, friendly, and informative in your responses.
        Format your responses clearly without any special tokens or unnecessary formatting characters."""
    
    # TTS now handled by Web Speech API in frontend
    # def setup_tts(self):
    #     """Configure text-to-speech settings"""
    #     voices = self.tts_engine.getProperty('voices')
    #     if voices:
    #         self.tts_engine.setProperty('voice', voices[0].id)
    #     self.tts_engine.setProperty('rate', 150)
    #     self.tts_engine.setProperty('volume', 0.8)
    
    def setup_vision_models(self):
        """Initialize vision and image processing models"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.huggingface_available = False
        self.nvidia_available = False
        
        # Initialize Hugging Face BLIP model with better error handling
        try:
            print("Loading Hugging Face BLIP model...")
            
            # Set cache directory for models
            cache_dir = os.path.join(os.getcwd(), "models_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Get Hugging Face token if available
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            
            # Try different BLIP model variants for better compatibility
            model_variants = [
                "Salesforce/blip-image-captioning-base",
                "Salesforce/blip-image-captioning-large"
            ]
            
            for model_name in model_variants:
                try:
                    print(f"   Trying {model_name}...")
                    
                    # Load processor and model
                    # Use `use_auth_token` for wider compatibility across transformer versions
                    self.blip_processor = BlipProcessor.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        use_auth_token=hf_token if hf_token else None,
                        trust_remote_code=False
                    )
                    
                    self.blip_model = BlipForConditionalGeneration.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                        use_auth_token=hf_token if hf_token else None,
                        device_map='auto',
                        low_cpu_mem_usage=True,
                        trust_remote_code=False
                    )
                    
                    # Move to device and set to evaluation mode
                    self.blip_model.to(self.device)
                    self.blip_model.eval()
                    
                    self.huggingface_available = True
                    print(f"✅ Hugging Face BLIP model loaded: {model_name}")
                    print(f"   Device: {self.device}")
                    break
                    
                except Exception as e:
                    print(f"   Failed to load {model_name}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error loading Hugging Face BLIP model: {e}")
            self.blip_processor = None
            self.blip_model = None
        
        # Initialize NVIDIA Vision API
        try:
            self.nvidia_api_key = os.getenv('NVIDIA_API_KEY')
            if self.nvidia_api_key:
                self.nvidia_available = True
                print("✅ NVIDIA Vision API key found")
            else:
                print("⚠️  NVIDIA_API_KEY not found in environment variables")
        except Exception as e:
            print(f"❌ Error setting up NVIDIA Vision API: {e}")
        
        # Check if any vision model is available
        if not self.huggingface_available and not self.nvidia_available:
            print("❌ No vision models available! Please check your API keys and model installations.")
        else:
            print(f"✅ Vision system ready - HuggingFace: {self.huggingface_available}, NVIDIA: {self.nvidia_available}")
    
    def setup_object_detection(self):
        """Initialize object detection model"""
        try:
            # Try to load YOLOv5 with better error handling
            print("Loading object detection model...")
            self.object_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False, verbose=False)
            self.object_detector.to(self.device)
            self.object_detector.eval()  # Set to evaluation mode
            print("✅ Object detection model loaded successfully")
        except Exception as e:
            print(f"⚠️  Error loading object detection model: {e}")
            print("   Vision mode will work with image captioning only (no object detection)")
            self.object_detector = None
    
    # Audio processing now handled by Web Speech API in frontend
    # def speech_to_text(self, audio_data):
    #     """Convert speech to text"""
    #     try:
    #         with sr.Microphone() as source:
    #             self.recognizer.adjust_for_ambient_noise(source)
    #             audio = self.recognizer.listen(source, timeout=5)
    #             text = self.recognizer.recognize_google(audio)
    #             return text
    #     except Exception as e:
    #         return f"Speech recognition error: {e}"
    
    # def text_to_speech(self, text):
    #     """Convert text to speech"""
    #     try:
    #         self.tts_engine.say(text)
    #         self.tts_engine.runAndWait()
    #     except Exception as e:
    #         print(f"TTS error: {e}")
    
    def web_search(self, query):
        """Perform web search using Tavily"""
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=3
            )
            return response.get('results', [])
        except Exception as e:
            print(f"Web search error: {e}")
            return []
    
    def process_image_with_blip(self, image):
        """Process image with BLIP for captioning"""
        try:
            # Check if BLIP model is available
            if not self.huggingface_available or self.blip_processor is None or self.blip_model is None:
                return "Hugging Face BLIP model not available"
            
            # Ensure image is in PIL format
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif not isinstance(image, Image.Image):
                image = Image.open(image)
            
            # Resize image if too large (BLIP has memory constraints)
            max_size = 512
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Process with BLIP
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption with better parameters
            with torch.no_grad():  # Save memory
                out = self.blip_model.generate(
                    **inputs, 
                    max_length=50,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # Clean up the caption
            caption = caption.strip()
            if not caption:
                caption = "Unable to generate caption for this image"
            
            return caption
            
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory for BLIP processing")
            return "Error: Not enough GPU memory for image processing"
        except Exception as e:
            print(f"BLIP processing error: {e}")
            return f"Error processing image with BLIP: {str(e)}"
    
    def process_image_with_nvidia(self, image):
        """Process image with NVIDIA Vision API"""
        try:
            # Convert image to base64
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                pil_image = Image.open(image)
            
            # Resize image if too large (NVIDIA API has size limits)
            max_size = 1024
            if pil_image.width > max_size or pil_image.height > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Prepare API request
            invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.nvidia_api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta/llama-3.2-90b-vision-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe what you see in this image in detail. Be specific about objects, colors, and any interesting details."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            # Make API request
            response = requests.post(invoke_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            caption = result['choices'][0]['message']['content']
            
            return caption
            
        except Exception as e:
            print(f"NVIDIA Vision API error: {e}")
            return f"Error processing image with NVIDIA Vision API: {str(e)}"
    
    def process_image_vision(self, image):
        """Process image using available vision models with fallback"""
        results = {
            'caption': '',
            'model_used': '',
            'huggingface_available': self.huggingface_available,
            'nvidia_available': self.nvidia_available
        }
        
        # Try NVIDIA first (usually better quality)
        if self.nvidia_available:
            try:
                print("Processing image with NVIDIA Vision API...")
                caption = self.process_image_with_nvidia(image)
                results['caption'] = caption
                results['model_used'] = 'nvidia'
                print("✅ NVIDIA Vision API processing successful")
                return results
            except Exception as e:
                print(f"❌ NVIDIA Vision API failed: {e}")
        
        # Fallback to Hugging Face BLIP
        if self.huggingface_available:
            try:
                print("Processing image with Hugging Face BLIP...")
                caption = self.process_image_with_blip(image)
                results['caption'] = caption
                results['model_used'] = 'huggingface'
                print("✅ Hugging Face BLIP processing successful")
                return results
            except Exception as e:
                print(f"❌ Hugging Face BLIP failed: {e}")
        
        # No models available
        results['caption'] = "No vision models available. Please check your API keys and model installations."
        results['model_used'] = 'none'
        return results
    
    def detect_objects(self, image):
        """Detect objects in image and return bounding boxes"""
        try:
            if self.object_detector is None:
                return []
            
            # Convert image for YOLO
            if isinstance(image, np.ndarray):
                results = self.object_detector(image)
            else:
                results = self.object_detector(np.array(image))
            
            # Extract detections
            detections = []
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                if conf > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, box)
                    class_name = self.object_detector.names[int(cls)]
                    # Compute normalized bounding box (x, y, w, h) relative to image dimensions (0-1 range)
                    if isinstance(image, np.ndarray):
                        h, w = image.shape[:2]
                    else:
                        h, w = np.array(image).shape[:2]
                    width = x2 - x1
                    height = y2 - y1
                    bbox = [x1 / w, y1 / h, width / w, height / h]

                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'bbox': bbox,
                        'confidence': float(conf),
                        'class': class_name
                    })
            
            return detections
        except Exception as e:
            print(f"Object detection error: {e}")
            return []
    
    def detect_objects_with_descriptions(self, image, scene_description):
        """Enhanced object detection with individual descriptions and random colors"""
        try:
            detections = self.detect_objects(image)
            if not detections:
                return []
            
            enhanced_detections = []
            for i, detection in enumerate(detections):
                class_name = detection['class']
                
                # Generate random color for each object
                import random
                color = (
                    random.randint(50, 255),
                    random.randint(50, 255), 
                    random.randint(50, 255)
                )
                
                # Use GPT OSS 120B to extract specific description for this object
                specific_description = self.extract_object_description_with_gpt(scene_description, class_name, image)
                
                # Store/update object in knowledge database
                self.store_detected_object(class_name, specific_description, detection['confidence'])
                
                # Queue for info gathering if new or rarely seen
                obj_info = self.knowledge_db.get_object(class_name)
                if not obj_info or obj_info.get('times_seen', 0) <= 3:
                    self.agentic_system.info_gatherer.queue_object_for_research(class_name, specific_description)
                
                enhanced_detections.append({
                    **detection,
                    'color': color,
                    'description': specific_description,
                    'id': f"obj_{i}",
                    'knowledge': obj_info
                })
            
            return enhanced_detections
        except Exception as e:
            print(f"Enhanced object detection error: {e}")
            return []
    
    def store_detected_object(self, object_name: str, description: str, confidence: float):
        """Store detected object in knowledge database"""
        try:
            self.knowledge_db.add_object(
                name=object_name,
                description=description,
                confidence_score=confidence
            )
        except Exception as e:
            print(f"Error storing object: {e}")
    
    def extract_object_description_with_gpt(self, scene_description, object_class, image):
        """Use GPT OSS 120B to extract specific description for an object"""
        try:
            # Create a focused prompt for object description
            prompt = f"""
            Based on this scene description: "{scene_description}"
            
            Focus specifically on the {object_class} in the scene. Provide a detailed, specific description of just this {object_class}, including:
            - Its appearance, color, and condition
            - Its position or orientation in the scene
            - Any notable features or details
            - Its context within the overall scene
            
            Be specific and detailed. Only describe the {object_class}, not other objects.
            """
            
            # Use GPT to get a focused description
            messages = [{"role": "user", "content": prompt}]
            description = self.chat_with_groq(messages, model="openai/gpt-oss-120b")
            
            return description.strip() if description else f"A {object_class} is visible in the scene."
            
        except Exception as e:
            print(f"GPT description extraction error: {e}")
            return f"A {object_class} is visible in the scene."
    
    def extract_object_description(self, scene_description, object_class):
        """Fallback method for object description extraction"""
        try:
            # Simple keyword-based extraction as fallback
            sentences = scene_description.split('.')
            
            for sentence in sentences:
                if object_class.lower() in sentence.lower():
                    return sentence.strip()
            
            return f"A {object_class} is visible in the scene."
        except Exception as e:
            print(f"Description extraction error: {e}")
            return f"A {object_class} is visible in the scene."
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on image with colors"""
        try:
            for detection in detections:
                x1, y1, x2, y2 = detection['box']
                class_name = detection['class']
                confidence = detection['confidence']
                color = detection.get('color', (0, 255, 0))  # Default green
                
                # Draw bounding box with specific color
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with same color
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return image
        except Exception as e:
            print(f"Drawing error: {e}")
            return image
    
    def clean_response_text(self, text):
        """Clean AI response text from unwanted formatting characters"""
        if not text:
            return ""
        
        # Remove common special tokens
        text = re.sub(r'<\|.*?\|>', '', text)  # Remove tokens like <|end|>
        text = re.sub(r'\[\[.*?\]\]', '', text)  # Remove [[tokens]]
        text = re.sub(r'<<.*?>>', '', text)  # Remove <<tokens>>
        
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Replace 3+ newlines with 2
        text = re.sub(r' {2,}', ' ', text)  # Replace multiple spaces with single
        
        # Remove any remaining special characters at start/end
        text = text.strip()
        
        # Unescape HTML entities if any
        text = html.unescape(text)
        
        return text

    def chat_with_groq(self, messages, model="openai/gpt-oss-120b", think_mode=False, reasoning_level="medium", custom_behavior="", temperature=None, stream=False):
        """Chat with Groq API with optional think mode and custom behavior"""
        try:
            # Add system prompt
            system_prompt = self.system_prompt
            formatted_messages = [{"role": "system", "content": system_prompt}]
            formatted_messages.extend(messages)
            
            # Add custom behavior if provided
            if custom_behavior:
                formatted_messages[0]["content"] += f"\n\nCustom Behavior Instructions: {custom_behavior}"
            
            # Modify system prompt for think mode
            if think_mode:
                think_prompts = {
                    "high": "Think step by step. Show your reasoning process in detail. Break down complex problems into smaller parts. Explain your thought process for each step.",
                    "medium": "Think through this step by step. Show your reasoning process. Explain your approach and key decisions.",
                    "low": "Think briefly about this. Show your main reasoning steps."
                }
                formatted_messages[0]["content"] += f"\n\n{think_prompts.get(reasoning_level, think_prompts['medium'])}"
            
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                max_tokens=4096,  # Increased token limit
                temperature=temperature if temperature is not None else self.chat_temperature,
                stream=stream
            )
            
            if stream:
                return response
            else:
                content = response.choices[0].message.content
                # Clean the response text
                return self.clean_response_text(content)
        except Exception as e:
            return f"Error: {e}"
    
    def process_file(self, file_content, file_type):
        """Process uploaded files"""
        try:
            if file_type.startswith('image/'):
                # Process image file
                image = Image.open(io.BytesIO(file_content))
                vision_results = self.process_image_vision(image)
                detections = self.detect_objects(image)
                
                return {
                    'type': 'image',
                    'caption': vision_results['caption'],
                    'model_used': vision_results['model_used'],
                    'huggingface_available': vision_results['huggingface_available'],
                    'nvidia_available': vision_results['nvidia_available'],
                    'detections': detections
                }
            
            elif file_type == 'text/plain':
                # Process text file
                text_content = file_content.decode('utf-8')
                return {
                    'type': 'text',
                    'content': text_content[:1000]  # Limit text length
                }
            
            else:
                return {'type': 'unsupported', 'message': 'File type not supported'}
                
        except Exception as e:
            return {'type': 'error', 'message': str(e)}

# Initialize AI Assistant
ai_assistant = AIAssistant()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        chat_history = data.get('history', [])
        web_search_enabled = data.get('web_search_enabled', True)
        think_mode = data.get('think_mode', False)
        reasoning_level = data.get('reasoning_level', 'medium')
        custom_behavior = data.get('custom_behavior', '')
        temperature = float(data.get('temperature', ai_assistant.chat_temperature))
        files = data.get('files', [])
        continue_generation = data.get('continue_generation', False)
        
        # Handle continuation
        if continue_generation:
            user_message = "Please continue from where you left off."
        
        # Check for web search request only if enabled
        if web_search_enabled and not continue_generation and any(keyword in user_message.lower() for keyword in ['search', 'look up', 'find information', 'what is', 'how to']):
            search_results = ai_assistant.web_search(user_message)
            if search_results:
                context = "\n".join([f"- {result.get('title', '')}: {result.get('content', '')}" 
                                   for result in search_results[:3]])
                user_message += f"\n\nSearch results:\n{context}"
        
        # Prepare messages for Groq
        messages = []
        for msg in chat_history[-10:]:  # Keep last 10 messages for context
            messages.append({"role": msg['role'], "content": msg['content']})
        
        if not continue_generation:
            messages.append({"role": "user", "content": user_message})
        
        # Get response from Groq with think mode support
        response = ai_assistant.chat_with_groq(messages, think_mode=think_mode, reasoning_level=reasoning_level, custom_behavior=custom_behavior, temperature=temperature)
        
        # Check if response was cut off (ends abruptly)
        needs_continuation = len(response) > 3800  # Near token limit
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'think_mode': think_mode,
            'reasoning_level': reasoning_level,
            'needs_continuation': needs_continuation
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Audio endpoints removed - now handled by Web Speech API in frontend
# @app.route('/speech-to-text', methods=['POST'])
# def speech_to_text():
#     try:
#         # This would be implemented with real-time audio capture
#         # For now, return a placeholder
#         return jsonify({'text': 'Speech recognition not implemented in demo'})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/text-to-speech', methods=['POST'])
# def text_to_speech():
#     try:
#         data = request.json
#         text = data.get('text', '')
        
#         # Run TTS in background thread to avoid blocking
#         threading.Thread(target=ai_assistant.text_to_speech, args=(text,)).start()
        
#         return jsonify({'status': 'success'})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        if file:
            file_content = file.read()
            file_type = file.content_type
            
            result = ai_assistant.process_file(file_content, file_type)
            return jsonify(result)
        else:
            return jsonify({'error': 'No file uploaded'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/start-vision', methods=['POST'])
def start_vision():
    try:
        ai_assistant.vision_mode = True
        
        # Try to initialize camera with better error handling
        if ai_assistant.camera is None:
            ai_assistant.camera = cv2.VideoCapture(0)
            
        # Check if camera is working
        if not ai_assistant.camera.isOpened():
            ai_assistant.camera = cv2.VideoCapture(0)
            
        # Prefer lower resolution to reduce latency
        try:
            ai_assistant.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            ai_assistant.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception:
            pass
        
        # Test camera
        ret, frame = ai_assistant.camera.read()
        if not ret:
            return jsonify({'error': 'Camera not accessible. Please check if camera is connected and not being used by another application.'}), 400
            
        return jsonify({'status': 'Vision mode started', 'camera_working': True})
    except Exception as e:
        return jsonify({'error': f'Failed to start vision mode: {str(e)}'}), 500

@app.route('/stop-vision', methods=['POST'])
def stop_vision():
    try:
        ai_assistant.vision_mode = False
        if ai_assistant.camera:
            ai_assistant.camera.release()
        return jsonify({'status': 'Vision mode stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video-feed')
def video_feed():
    def generate_frames():
        while ai_assistant.vision_mode and ai_assistant.camera and ai_assistant.camera.isOpened():
            try:
                success, frame = ai_assistant.camera.read()
                if not success:
                    print("Failed to read frame from camera")
                    break
                
                # Store current frame for analysis
                with ai_assistant.frame_lock:
                    ai_assistant.current_frame = frame.copy()
                
                # Just show the raw frame without processing (for performance)
                # Processing will happen only when analyze button is clicked
                
                # Convert frame to bytes
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ret:
                    print("Failed to encode frame")
                    break
                    
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error in video feed: {e}")
                break
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    try:
        # Use stored current frame instead of reading new one (thread-safe)
        with ai_assistant.frame_lock:
            if ai_assistant.current_frame is None:
                if not ai_assistant.camera or not ai_assistant.camera.isOpened():
                    return jsonify({'error': 'Camera not available'}), 400
                success, frame = ai_assistant.camera.read()
                if not success:
                    return jsonify({'error': 'Failed to capture frame'}), 500
                ai_assistant.current_frame = frame
            frame = ai_assistant.current_frame
            ai_assistant.current_frame = None
        
        # Process image with available vision models
        vision_results = ai_assistant.process_image_vision(frame)
        
        # Get enhanced object detections with descriptions and colors
        detections = []
        if ai_assistant.object_detector is not None:
            detections = ai_assistant.detect_objects_with_descriptions(frame, vision_results['caption'])
        
        # current_frame cleared in locked section above
        
        return jsonify({
            'caption': vision_results['caption'],
            'model_used': vision_results['model_used'],
            'huggingface_available': vision_results['huggingface_available'],
            'nvidia_available': vision_results['nvidia_available'],
            'detections': detections,
            'object_count': len(detections),
            'object_detection_available': ai_assistant.object_detector is not None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/vision-status', methods=['GET'])
def vision_status():
    """Get status of all vision models"""
    try:
        return jsonify({
            'huggingface_available': ai_assistant.huggingface_available,
            'nvidia_available': ai_assistant.nvidia_available,
            'object_detection_available': ai_assistant.object_detector is not None,
            'device': str(ai_assistant.device),
            'models_loaded': {
                'huggingface_blip': ai_assistant.huggingface_available,
                'nvidia_vision_api': ai_assistant.nvidia_available,
                'yolo_object_detection': ai_assistant.object_detector is not None
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask-about-object', methods=['POST'])
def ask_about_object():
    """Answer questions about a specific detected object using web search"""
    try:
        data = request.json
        object_class = data.get('object_class', '')
        object_description = data.get('object_description', '')
        user_question = data.get('question', '')
        location = data.get('location', 'India')
        
        if not object_class or not user_question:
            return jsonify({'error': 'Object class and question are required'}), 400
        
        # Parse the user question to understand what they're asking
        question_lower = user_question.lower()
        
        # Determine the type of query
        search_query = ""
        if 'price' in question_lower or 'cost' in question_lower:
            search_query = f"{object_class} price cost {location} current 2024 market rate"
        elif 'where' in question_lower and ('grow' in question_lower or 'cultivate' in question_lower):
            search_query = f"{object_class} cultivation growing regions {location} farming conditions climate"
        elif 'condition' in question_lower or 'climate' in question_lower:
            search_query = f"{object_class} growing conditions climate requirements soil temperature rainfall"
        elif 'nutrition' in question_lower or 'benefit' in question_lower or 'health' in question_lower:
            search_query = f"{object_class} nutritional value health benefits vitamins minerals calories"
        elif 'buy' in question_lower or 'purchase' in question_lower or 'shop' in question_lower:
            search_query = f"{object_class} where to buy {location} online shopping stores market"
        elif 'recipe' in question_lower or 'cook' in question_lower or 'prepare' in question_lower:
            search_query = f"{object_class} recipes cooking methods preparation dishes {location} cuisine"
        else:
            # General query
            search_query = f"{object_class} {user_question} {location}"
        
        # Perform web search
        search_results = ai_assistant.web_search(search_query)
        
        # Prepare context from search results
        search_context = "\n".join([
            f"- {result.get('title', '')}: {result.get('content', '')[:300]}"
            for result in search_results[:3]
        ])
        
        # Generate comprehensive answer using GPT
        answer_prompt = f"""
        Object: {object_class}
        Description from image: {object_description}
        User Question: {user_question}
        Location: {location}
        
        Web Search Results:
        {search_context}
        
        Please provide a detailed, accurate answer to the user's question based on the search results.
        Include specific information like prices, locations, conditions, etc. if available.
        Be factual and cite information from the search results.
        If the search results don't contain enough information, acknowledge this.
        """
        
        messages = [{"role": "user", "content": answer_prompt}]
        answer = ai_assistant.chat_with_groq(messages, model="openai/gpt-oss-120b")
        
        return jsonify({
            'object': object_class,
            'question': user_question,
            'answer': answer,
            'search_query': search_query,
            'sources': [{
                'title': result.get('title', ''),
                'url': result.get('url', '')
            } for result in search_results[:3]]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def terminal_interface():
    """Terminal interface for the AI assistant"""
    print("🤖 Moi AI Assistant - Terminal Interface")
    print("Model: Groq OpenAI GPT-OSS 120B")
    print("Vision: Salesforce/blip-image-captioning-base")
    print("Audio: Web Speech API (browser only)")
    print("Commands: 'exit' to quit, 'search: <query>' for web search")
    print("-" * 60)
    
    chat_history = []
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'exit':
                print("👋 Goodbye!")
                break
            
            if user_input.startswith('search:'):
                query = user_input[7:].strip()
                print("🔍 Searching...")
                search_results = ai_assistant.web_search(query)
                
                if search_results:
                    print("\n📊 Search Results:")
                    for i, result in enumerate(search_results[:3], 1):
                        print(f"{i}. {result.get('title', 'No title')}")
                        print(f"   {result.get('content', 'No content')[:200]}...")
                        print()
                continue
            
            # Add to chat history
            chat_history.append({"role": "user", "content": user_input})
            
            print("🔄 Processing with GPT-OSS 120B...")
            
            # Get response
            messages = chat_history[-10:]  # Keep last 10 messages
            response = ai_assistant.chat_with_groq(messages, model="openai/gpt-oss-120b")
            
            print(f"\n🤖 Moi: {response}")
            
            # Add response to history
            chat_history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

@app.route('/improve-prompt', methods=['POST'])
def improve_prompt():
    """Improve user prompt using AI"""
    try:
        data = request.json
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Use Groq to improve the prompt
        response = ai_assistant.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert at improving prompts to get better AI responses. Your task is to enhance the user's prompt to be more specific, detailed, and structured. Do not add unnecessary complexity, but make it clearer and more likely to get a high-quality response. Return ONLY the improved prompt without explanations or additional text."},
                {"role": "user", "content": f"Improve this prompt: {prompt}"}
            ],
            model="openai/gpt-oss-120b",
            temperature=0.5,
            max_tokens=500
        )
        
        improved_prompt = ai_assistant.clean_response_text(response.choices[0].message.content.strip())
        return jsonify({'improved_prompt': improved_prompt})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search-object', methods=['POST'])
def search_object():
    """Search web for object information with context-aware queries"""
    try:
        data = request.json
        object_name = data.get('object_name', '')
        query_context = data.get('query_context', '')  # Additional context from user query
        location = data.get('location', 'India')  # Default location
        
        if not object_name:
            return jsonify({'error': 'No object name provided'}), 400
        
        # Build context-aware search query
        if query_context:
            # User has specific questions about the object
            query = f"{object_name} {query_context} in {location}"
        else:
            # Default comprehensive search
            query = f"{object_name} current price {location} where to buy specifications features reviews 2024"
        
        # Perform web search
        search_results = ai_assistant.web_search(query)
        
        # Format and enhance results
        formatted_results = []
        for result in search_results[:5]:
            formatted_results.append({
                'title': result.get('title', ''),
                'content': result.get('content', ''),
                'url': result.get('url', ''),
                'score': result.get('score', 0)  # Relevance score if available
            })
        
        # Use GPT to summarize findings if we have results
        summary = ""
        if formatted_results:
            summary_prompt = f"""Based on these search results about {object_name}:
            {' '.join([r['content'][:200] for r in formatted_results[:3]])}
            
            Provide a brief summary including:
            - Current price range in {location}
            - Where it can be purchased
            - Key features or characteristics
            
            Keep it concise and factual."""
            
            messages = [{"role": "user", "content": summary_prompt}]
            summary = ai_assistant.chat_with_groq(messages, model="openai/gpt-oss-120b")
        
        return jsonify({
            'object': object_name,
            'location': location,
            'query': query,
            'summary': summary,
            'results': formatted_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========== NEW ENDPOINTS FOR ENHANCED FEATURES ==========

@app.route('/generate-image', methods=['POST'])
def generate_image():
    """Generate image from text prompt"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        size = data.get('size', '512x512')
        backend = data.get('backend', 'auto')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Generate image
        image_base64 = ai_assistant.image_generator.generate_and_encode(prompt, backend, size)
        
        if image_base64:
            return jsonify({
                'success': True,
                'image': image_base64,
                'prompt': prompt,
                'backend': backend
            })
        else:
            return jsonify({'error': 'Image generation failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge/objects', methods=['GET'])
def get_all_objects():
    """Get all objects from knowledge database"""
    try:
        limit = int(request.args.get('limit', 100))
        objects = ai_assistant.knowledge_db.get_all_objects(limit=limit)
        return jsonify({'objects': objects, 'count': len(objects)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge/object/<object_name>', methods=['GET'])
def get_object_info(object_name):
    """Get detailed information about a specific object"""
    try:
        obj_info = ai_assistant.knowledge_db.get_object(object_name)
        if obj_info:
            return jsonify(obj_info)
        else:
            return jsonify({'error': 'Object not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge/object', methods=['POST'])
def add_or_update_object():
    """Add or update object information (User Teaching Interface)"""
    try:
        data = request.json
        object_name = data.get('name', '')
        
        if not object_name:
            return jsonify({'error': 'Object name is required'}), 400
        
        # Extract fields
        description = data.get('description', '')
        color = data.get('color', '')
        climate = data.get('climate', '')
        types = data.get('types', '')
        category = data.get('category', '')
        user_notes = data.get('user_notes', '')
        
        # Add or update object
        object_id = ai_assistant.knowledge_db.add_object(
            name=object_name,
            description=description,
            color=color,
            climate=climate,
            types=types,
            category=category,
            user_notes=user_notes
        )
        
        # Add custom properties if provided
        properties = data.get('properties', {})
        for key, value in properties.items():
            ai_assistant.knowledge_db.add_property(object_name, key, value, source='user')
        
        return jsonify({
            'success': True,
            'object_id': object_id,
            'message': f'Object "{object_name}" saved successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge/object/<object_name>', methods=['DELETE'])
def delete_object(object_name):
    """Delete object from knowledge database"""
    try:
        ai_assistant.knowledge_db.delete_object(object_name)
        return jsonify({'success': True, 'message': f'Object "{object_name}" deleted'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge/search', methods=['GET'])
def search_knowledge():
    """Search knowledge database"""
    try:
        query = request.args.get('query', '')
        category = request.args.get('category', '')
        limit = int(request.args.get('limit', 10))
        
        results = ai_assistant.knowledge_db.search_objects(
            query=query if query else None,
            category=category if category else None,
            limit=limit
        )
        
        return jsonify({'results': results, 'count': len(results)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge/statistics', methods=['GET'])
def get_knowledge_statistics():
    """Get knowledge database statistics"""
    try:
        stats = ai_assistant.knowledge_db.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge/export', methods=['GET'])
def export_knowledge():
    """Export knowledge database to JSON"""
    try:
        file_path = ai_assistant.knowledge_db.export_knowledge()
        return jsonify({
            'success': True,
            'file_path': file_path,
            'message': 'Knowledge exported successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/agentic/status', methods=['GET'])
def get_agentic_status():
    """Get status of all agentic AI systems"""
    try:
        status = ai_assistant.agentic_system.get_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/agentic/secret-model/generate', methods=['POST'])
def secret_model_generate():
    """Generate response using the secret model"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 100)
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        response = ai_assistant.agentic_system.secret_trainer.generate_response(prompt, max_length)
        
        return jsonify({
            'prompt': prompt,
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat-with-memory', methods=['POST'])
def chat_with_memory():
    """Enhanced chat that uses knowledge database for context"""
    try:
        data = request.json
        user_message = data.get('message', '')
        chat_history = data.get('history', [])
        
        # Extract potential object names from message
        words = user_message.lower().split()
        context_info = []
        
        for word in words:
            obj_info = ai_assistant.knowledge_db.get_object(word)
            if obj_info:
                context_info.append(f"I know about {word}: {obj_info.get('description', '')}")
        
        # Add context to message
        if context_info:
            enhanced_message = f"{user_message}\n\n[Context from my memory: {' '.join(context_info)}]"
        else:
            enhanced_message = user_message
        
        # Prepare messages
        messages = []
        for msg in chat_history[-10:]:
            messages.append({"role": msg['role'], "content": msg['content']})
        messages.append({"role": "user", "content": enhanced_message})
        
        # Get response
        response = ai_assistant.chat_with_groq(messages)
        
        # Store conversation as training data
        ai_assistant.knowledge_db.add_training_data(
            input_text=user_message,
            output_text=response,
            context=json.dumps(chat_history[-3:]),
            quality_score=0.7
        )
        
        return jsonify({
            'response': response,
            'context_used': len(context_info) > 0,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'terminal':
        # Run terminal interface
        terminal_interface()
    else:
        # Run Flask web interface
        print("🚀 Starting Moi AI Assistant...")
        print("🌐 Web interface: http://localhost:5000")
        print("💻 Terminal interface: python app.py terminal")
        app.run(debug=True, host='0.0.0.0', port=5000)

