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

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

class AIAssistant:
    def __init__(self):
        # Initialize API clients
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
        
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
        
        # Knowledge DB (JSON-backed)
        self.knowledge_db_path = os.path.join(os.getcwd(), "data", "knowledge_db.json")
        os.makedirs(os.path.dirname(self.knowledge_db_path), exist_ok=True)
        self.knowledge_lock = threading.Lock()
        self.knowledge = self.load_knowledge_db()

        # Background agents
        self.agents_enabled = os.getenv('BACKGROUND_AGENTS', 'true').lower() in ['1', 'true', 'yes']
        self.agent_interval_seconds = int(os.getenv('AGENT_INTERVAL_SECONDS', '120'))
        self.shutdown_event = threading.Event()
        self.secret_model_state_path = os.path.join(os.getcwd(), "data", "secret_model_state.json")
        if self.agents_enabled:
            self.start_background_agents()
        
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
                
                enhanced_detections.append({
                    **detection,
                    'color': color,
                    'description': specific_description,
                    'id': f"obj_{i}"
                })
            
            return enhanced_detections
        except Exception as e:
            print(f"Enhanced object detection error: {e}")
            return []
    
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

    # ==========================
    # Knowledge DB functionality
    # ==========================
    def normalize_object_name(self, name):
        try:
            return re.sub(r'\s+', ' ', str(name).strip().lower())
        except Exception:
            return str(name).strip().lower() if name else ''

    def load_knowledge_db(self):
        try:
            if os.path.exists(self.knowledge_db_path):
                with open(self.knowledge_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Ensure keys are normalized
                    normalized = {}
                    for key, value in data.items():
                        normalized[self.normalize_object_name(key)] = value
                    return normalized
            return {}
        except Exception as e:
            print(f"Knowledge DB load error: {e}")
            return {}

    def save_knowledge_db(self):
        try:
            with self.knowledge_lock:
                with open(self.knowledge_db_path, 'w', encoding='utf-8') as f:
                    json.dump(self.knowledge, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Knowledge DB save error: {e}")

    def get_object_knowledge(self, name):
        try:
            key = self.normalize_object_name(name)
            return self.knowledge.get(key)
        except Exception:
            return None

    def upsert_object_knowledge(self, name, data):
        try:
            key = self.normalize_object_name(name)
            if not key:
                raise ValueError('Invalid object name')
            with self.knowledge_lock:
                existing = self.knowledge.get(key, {})
                # Merge shallowly
                merged = {**existing, **(data or {})}
                # Initialize structured fields
                if 'user_notes' not in merged or not isinstance(merged.get('user_notes'), list):
                    merged['user_notes'] = existing.get('user_notes', []) if isinstance(existing.get('user_notes'), list) else []
                # Maintain canonical name
                merged['name'] = merged.get('name') or name
                merged['updated_at'] = datetime.utcnow().isoformat()
                # Save
                self.knowledge[key] = merged
                self.save_knowledge_db()
                return merged
        except Exception as e:
            raise e

    def add_note_to_object(self, name, note):
        try:
            key = self.normalize_object_name(name)
            if not key:
                raise ValueError('Invalid object name')
            with self.knowledge_lock:
                entry = self.knowledge.get(key, {'name': name, 'user_notes': []})
                if 'user_notes' not in entry or not isinstance(entry['user_notes'], list):
                    entry['user_notes'] = []
                entry['user_notes'].append({
                    'note': str(note),
                    'timestamp': datetime.utcnow().isoformat()
                })
                entry['updated_at'] = datetime.utcnow().isoformat()
                self.knowledge[key] = entry
                self.save_knowledge_db()
                return entry
        except Exception as e:
            raise e

    def delete_object_knowledge(self, name):
        try:
            key = self.normalize_object_name(name)
            with self.knowledge_lock:
                existed = key in self.knowledge
                if existed:
                    del self.knowledge[key]
                    self.save_knowledge_db()
                return existed
        except Exception as e:
            raise e

    def search_knowledge(self, query, limit=20):
        try:
            q = self.normalize_object_name(query)
            results = []
            with self.knowledge_lock:
                for key, entry in self.knowledge.items():
                    if q in key:
                        results.append(entry)
                        continue
                    # Search in selected fields
                    haystack = ' '.join([
                        str(entry.get('name', '')),
                        str(entry.get('color', '')),
                        str(entry.get('types', '')),
                        str(entry.get('climate', '')),
                        str(entry.get('category', '')),
                        str(entry.get('description', ''))
                    ]).lower()
                    if q and q in haystack:
                        results.append(entry)
                    if len(results) >= limit:
                        break
            return results
        except Exception as e:
            print(f"Knowledge search error: {e}")
            return []

    # =============================
    # Knowledge enrichment pipeline
    # =============================
    def auto_enrich_object_knowledge(self, object_name, scene_caption=""):
        try:
            object_name_norm = self.normalize_object_name(object_name)
            if not object_name_norm:
                return
            # Avoid duplicate enrich if already known
            if self.get_object_knowledge(object_name_norm):
                return
            # Perform targeted web search
            query = f"{object_name} facts color types climate nutrients uses description"
            results = self.web_search(query)
            context = "\n".join([f"- {r.get('title','')}: {r.get('content','')}" for r in results[:3]])
            # Ask LLM to produce structured JSON
            prompt = f"""
            You are a structured information extractor. Build a concise JSON object for the object named "{object_name}".
            Include fields if available: name, category, color, colors, types, climate, nutrients, features, description, synonyms.
            Use arrays for list-like fields. Keep values short and factual. If unknown, omit the field.
            Scene hint (may help disambiguate): {scene_caption}
            Evidence:
            {context}

            Return ONLY valid JSON.
            """
            messages = [{"role": "user", "content": prompt}]
            raw = self.chat_with_groq(messages, model="openai/gpt-oss-120b", temperature=0.2)
            data = {}
            try:
                data = json.loads(raw)
            except Exception:
                # Fallback minimal record
                data = {"name": object_name, "description": self.clean_response_text(raw)[:500]}
            # Ensure name present
            if 'name' not in data or not data['name']:
                data['name'] = object_name
            self.upsert_object_knowledge(object_name, data)
        except Exception as e:
            print(f"Auto enrich error for {object_name}: {e}")

    # =============================
    # Background Agents (Trainer & Supervisor)
    # =============================
    def start_background_agents(self):
        try:
            threading.Thread(target=self.agent1_trainer_loop, name='Agent1Trainer', daemon=True).start()
            threading.Thread(target=self.agent3_supervisor_loop, name='Agent3Supervisor', daemon=True).start()
            print("✅ Background agents started")
        except Exception as e:
            print(f"⚠️  Failed to start background agents: {e}")

    def read_secret_model_state(self):
        try:
            if os.path.exists(self.secret_model_state_path):
                with open(self.secret_model_state_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {"last_updated": None, "notes": [], "stats": {"train_iterations": 0}}
        except Exception as e:
            print(f"Secret model state read error: {e}")
            return {"last_updated": None, "notes": [], "stats": {"train_iterations": 0}}

    def write_secret_model_state(self, state):
        try:
            with open(self.secret_model_state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Secret model state write error: {e}")

    def agent1_trainer_loop(self):
        """Simulate training of hidden generative model using collected knowledge."""
        while not self.shutdown_event.is_set():
            try:
                state = self.read_secret_model_state()
                # Create a compact summary from knowledge as pseudo-training signal
                with self.knowledge_lock:
                    sample_items = list(self.knowledge.values())[:10]
                summary = "; ".join([
                    f"{item.get('name','unknown')}: {item.get('category','') or ''} {item.get('colors', item.get('color','')) or ''}"
                    for item in sample_items
                ])
                if summary:
                    note = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "event": "train_step",
                        "summary": summary[:500]
                    }
                    state.setdefault('notes', []).append(note)
                    stats = state.setdefault('stats', {"train_iterations": 0})
                    stats['train_iterations'] = int(stats.get('train_iterations', 0)) + 1
                    state['last_updated'] = datetime.utcnow().isoformat()
                    self.write_secret_model_state(state)
            except Exception as e:
                print(f"Agent1 trainer error: {e}")
            finally:
                time.sleep(self.agent_interval_seconds)

    def agent3_supervisor_loop(self):
        """Supervisor that decides which objects to enrich next and monitors availability of models."""
        while not self.shutdown_event.is_set():
            try:
                # Check model availability logs
                print(f"[Supervisor] Models - HF:{self.huggingface_available} NV:{self.nvidia_available} OD:{self.object_detector is not None}")
                # Identify objects missing fields and queue enrichment
                with self.knowledge_lock:
                    items = list(self.knowledge.values())
                for item in items[:5]:
                    needs = not item.get('types') or not item.get('description') or not item.get('colors')
                    if needs:
                        threading.Thread(target=self.auto_enrich_object_knowledge, args=(item.get('name',''), ''), daemon=True).start()
                # Also try to enrich recently detected common classes if unknown
                # (simple heuristic: common COCO classes)
                common_classes = ["person", "bicycle", "car", "dog", "cat", "orange", "apple", "chair", "bottle"]
                for cls in common_classes:
                    if not self.get_object_knowledge(cls):
                        threading.Thread(target=self.auto_enrich_object_knowledge, args=(cls, ''), daemon=True).start()
            except Exception as e:
                print(f"Agent3 supervisor error: {e}")
            finally:
                time.sleep(self.agent_interval_seconds)

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
        
        # Attach knowledge entries and trigger enrichment for unknowns
        enriched_detections = []
        for det in detections:
            obj_name = det.get('class', '')
            knowledge = ai_assistant.get_object_knowledge(obj_name)
            known = knowledge is not None
            det_with_knowledge = {**det, 'known': known, 'knowledge': knowledge}
            enriched_detections.append(det_with_knowledge)
            
            # Auto-enrich unknown objects in background
            if not known and os.getenv('AUTO_ENRICH', 'true').lower() in ['1','true','yes']:
                threading.Thread(target=ai_assistant.auto_enrich_object_knowledge, args=(obj_name, vision_results['caption']), daemon=True).start()
        
        detections = enriched_detections
        
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

# ==========================
# Knowledge DB API endpoints
# ==========================
@app.route('/knowledge', methods=['GET'])
def list_knowledge():
    try:
        q = request.args.get('q', '').strip()
        if q:
            items = ai_assistant.search_knowledge(q)
        else:
            with ai_assistant.knowledge_lock:
                items = list(ai_assistant.knowledge.values())
        return jsonify({'items': items, 'count': len(items)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge/search', methods=['GET'])
def search_knowledge_endpoint():
    try:
        q = request.args.get('q', '')
        items = ai_assistant.search_knowledge(q)
        return jsonify({'items': items, 'count': len(items)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge/<name>', methods=['GET'])
def get_knowledge(name):
    try:
        data = ai_assistant.get_object_knowledge(name)
        if not data:
            return jsonify({'error': 'Not found'}), 404
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge/<name>', methods=['PUT', 'PATCH'])
def upsert_knowledge(name):
    try:
        payload = request.get_json(force=True, silent=True) or {}
        updated = ai_assistant.upsert_object_knowledge(name, payload)
        return jsonify(updated)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge/<name>/notes', methods=['POST'])
def add_note_knowledge(name):
    try:
        payload = request.get_json(force=True, silent=True) or {}
        note = payload.get('note', '')
        if not note:
            return jsonify({'error': 'Note is required'}), 400
        entry = ai_assistant.add_note_to_object(name, note)
        return jsonify(entry)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge/<name>', methods=['DELETE'])
def delete_knowledge(name):
    try:
        existed = ai_assistant.delete_object_knowledge(name)
        if not existed:
            return jsonify({'status': 'not_found'}), 404
        return jsonify({'status': 'deleted'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================
# Image generation endpoint
# ==========================
@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        # Try Stability API first
        stability_key = os.getenv('STABILITY_API_KEY')
        if stability_key:
            try:
                resp = requests.post(
                    'https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image',
                    headers={
                        'Authorization': f'Bearer {stability_key}',
                        'Accept': 'application/json',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'text_prompts': [{'text': prompt}],
                        'cfg_scale': 7,
                        'clip_guidance_preset': 'FAST_BLUE',
                        'height': 512,
                        'width': 512,
                        'samples': 1,
                        'steps': 30
                    },
                    timeout=60
                )
                resp.raise_for_status()
                out = resp.json()
                if out.get('artifacts'):
                    b64 = out['artifacts'][0].get('base64')
                    return jsonify({'image_base64': b64, 'provider': 'stability'})
            except Exception as e:
                print(f"Stability API error: {e}")
        # Fallback to Hugging Face Inference API
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            try:
                resp = requests.post(
                    'https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1',
                    headers={
                        'Authorization': f'Bearer {hf_token}',
                        'Accept': 'image/png'
                    },
                    data=prompt.encode('utf-8'),
                    timeout=60
                )
                resp.raise_for_status()
                img_bytes = resp.content
                b64 = base64.b64encode(img_bytes).decode('utf-8')
                return jsonify({'image_base64': b64, 'provider': 'huggingface'})
            except Exception as e:
                print(f"HF Inference API error: {e}")
        return jsonify({'error': 'No image generation provider configured'}), 400
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

