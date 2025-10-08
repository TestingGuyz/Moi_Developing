"""
Moi AI Assistant - Fixed Headless Version
Eliminates OpenCV dependencies and adds graceful error handling
"""

import os
import json
import base64
import numpy as np
import requests
import re
import html
import sqlite3
import logging
import urllib.parse
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import io
from dotenv import load_dotenv
import threading
import time
from typing import Dict, List, Optional, Any

# Conditional imports with graceful handling
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not available - install with: pip install groq")

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("⚠️ Tavily not available - install with: pip install tavily-python")

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available - install with: pip install transformers torch")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("❌ PIL required - install with: pip install pillow")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# ==================== HEADLESS IMAGE PROCESSOR ====================

class HeadlessImageProcessor:
    """
    Headless image processing using only PIL/Pillow
    Replaces OpenCV functionality without system dependencies
    """
    
    def __init__(self):
        self.supported_formats = ['JPEG', 'PNG', 'WebP', 'BMP']
    
    def read_image_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """Read image from bytes using PIL"""
        try:
            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Error reading image from bytes: {e}")
            return None
    
    def read_image_from_file(self, file_path: str) -> Image.Image:
        """Read image from file using PIL"""
        try:
            return Image.open(file_path)
        except Exception as e:
            logger.error(f"Error reading image from file: {e}")
            return None
    
    def resize_image(self, image: Image.Image, size: tuple) -> Image.Image:
        """Resize image maintaining aspect ratio"""
        try:
            image.thumbnail(size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image
    
    def image_to_bytes(self, image: Image.Image, format: str = 'JPEG', quality: int = 85) -> bytes:
        """Convert PIL image to bytes"""
        try:
            buffer = io.BytesIO()
            if format.upper() == 'JPEG':
                # Convert RGBA to RGB for JPEG
                if image.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                    image = background
            
            image.save(buffer, format=format, quality=quality)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error converting image to bytes: {e}")
            return None
    
    def image_to_base64(self, image: Image.Image, format: str = 'JPEG') -> str:
        """Convert PIL image to base64 string"""
        try:
            image_bytes = self.image_to_bytes(image, format)
            if image_bytes:
                return base64.b64encode(image_bytes).decode('utf-8')
            return None
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return None
    
    def draw_bounding_box(self, image: Image.Image, bbox: tuple, label: str = "", color: tuple = (255, 0, 0)) -> Image.Image:
        """Draw bounding box on image using PIL"""
        try:
            draw = ImageDraw.Draw(image)
            x1, y1, x2, y2 = bbox
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label if provided
            if label:
                try:
                    # Try to use default font
                    font = ImageFont.load_default()
                except:
                    font = None
                
                # Calculate text position
                if font:
                    text_bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                else:
                    text_width, text_height = len(label) * 6, 11
                
                # Draw background rectangle for text
                draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
                # Draw text
                draw.text((x1 + 2, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
            
            return image
        except Exception as e:
            logger.error(f"Error drawing bounding box: {e}")
            return image

# ==================== HEADLESS CAMERA ====================

class HeadlessCamera:
    """
    Mock camera system for headless environments
    Generates test frames and handles camera simulation
    """
    
    def __init__(self):
        self.is_opened = False
        self.frame_count = 0
        self.test_mode = True
    
    def open(self) -> bool:
        """Simulate camera opening"""
        self.is_opened = True
        logger.info("📷 Headless camera simulation started")
        return True
    
    def isOpened(self) -> bool:
        """Check if camera is opened"""
        return self.is_opened
    
    def read(self) -> tuple:
        """Generate mock frame or return test image"""
        if not self.is_opened:
            return False, None
        
        try:
            # Generate a simple test image
            width, height = 640, 480
            image = Image.new('RGB', (width, height), color=(100, 150, 200))
            draw = ImageDraw.Draw(image)
            
            # Draw some test content
            draw.rectangle([50, 50, width-50, height-50], outline=(255, 255, 255), width=3)
            draw.text((width//2 - 100, height//2 - 10), f"Test Frame #{self.frame_count}", fill=(255, 255, 255))
            
            # Draw moving element
            x = (self.frame_count * 5) % (width - 100)
            draw.ellipse([x, height//2 + 50, x + 50, height//2 + 100], fill=(255, 255, 0))
            
            self.frame_count += 1
            
            # Convert PIL to numpy array (for compatibility)
            frame_array = np.array(image)
            
            return True, frame_array
            
        except Exception as e:
            logger.error(f"Error generating mock frame: {e}")
            return False, None
    
    def release(self):
        """Release camera resources"""
        self.is_opened = False
        logger.info("📷 Headless camera simulation stopped")
    
    def set(self, prop: int, value: int):
        """Mock camera property setting"""
        pass  # No-op for headless mode

# ==================== KNOWLEDGE DATABASE ====================

class KnowledgeDatabase:
    """Lightweight knowledge database for object information"""
    
    def __init__(self, db_path: str = "moi_knowledge.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS objects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        description TEXT,
                        category TEXT,
                        times_seen INTEGER DEFAULT 1,
                        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                ''')
                
                conn.commit()
                conn.close()
                logger.info("✅ Knowledge database initialized")
            except Exception as e:
                logger.error(f"Database initialization error: {e}")
    
    def add_object(self, name: str, description: str = "", category: str = "") -> int:
        """Add or update object in database"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check if exists
                cursor.execute('SELECT id, times_seen FROM objects WHERE name = ?', (name,))
                result = cursor.fetchone()
                
                if result:
                    # Update existing
                    object_id, times_seen = result
                    cursor.execute('''
                        UPDATE objects 
                        SET times_seen = ?, last_seen = ?, description = ?, category = ?
                        WHERE id = ?
                    ''', (times_seen + 1, datetime.now().isoformat(), description, category, object_id))
                else:
                    # Insert new
                    cursor.execute('''
                        INSERT INTO objects (name, description, category)
                        VALUES (?, ?, ?)
                    ''', (name, description, category))
                    object_id = cursor.lastrowid
                
                conn.commit()
                conn.close()
                return object_id
            except Exception as e:
                logger.error(f"Error adding object: {e}")
                return 0
    
    def get_object(self, name: str) -> Optional[Dict[str, Any]]:
        """Get object information"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM objects WHERE name = ?', (name,))
                result = cursor.fetchone()
                
                if result:
                    return dict(result)
                
                conn.close()
                return None
            except Exception as e:
                logger.error(f"Error getting object: {e}")
                return None
    
    def search_objects(self, query: str = "", limit: int = 10) -> List[Dict[str, Any]]:
        """Search objects"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if query:
                    cursor.execute('''
                        SELECT * FROM objects 
                        WHERE name LIKE ? OR description LIKE ?
                        ORDER BY times_seen DESC, last_seen DESC
                        LIMIT ?
                    ''', (f'%{query}%', f'%{query}%', limit))
                else:
                    cursor.execute('''
                        SELECT * FROM objects 
                        ORDER BY times_seen DESC, last_seen DESC
                        LIMIT ?
                    ''', (limit,))
                
                results = [dict(row) for row in cursor.fetchall()]
                conn.close()
                return results
            except Exception as e:
                logger.error(f"Error searching objects: {e}")
                return []

# ==================== IMAGE GENERATOR ====================

class HeadlessImageGenerator:
    """Headless image generation using free APIs"""
    
    def __init__(self):
        self.available_backends = ['pollinations', 'picsum']  # Free services
    
    def generate_image(self, prompt: str, size: str = '512x512') -> Optional[bytes]:
        """Generate image using free services"""
        try:
            width, height = map(int, size.split('x'))
            
            # Try Pollinations.ai first (free AI image generation)
            try:
                encoded_prompt = urllib.parse.quote(prompt)
                url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true"
                
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                return response.content
            except:
                pass
            
            # Fallback to Lorem Picsum (placeholder images)
            try:
                url = f"https://picsum.photos/{width}/{height}"
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                return response.content
            except:
                pass
            
            return None
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return None
    
    def generate_and_encode(self, prompt: str, size: str = '512x512') -> Optional[str]:
        """Generate image and return base64"""
        try:
            image_bytes = self.generate_image(prompt, size)
            if image_bytes:
                return base64.b64encode(image_bytes).decode('utf-8')
            return None
        except Exception as e:
            logger.error(f"Error generating and encoding image: {e}")
            return None

# ==================== AI ASSISTANT ====================

class HeadlessAIAssistant:
    """Main AI Assistant with headless compatibility"""
    
    def __init__(self):
        # Initialize components
        self.image_processor = HeadlessImageProcessor()
        self.knowledge_db = KnowledgeDatabase()
        self.image_generator = HeadlessImageGenerator()
        
        # Initialize API clients with error handling
        self.groq_client = None
        self.tavily_client = None
        
        if GROQ_AVAILABLE:
            try:
                groq_key = os.getenv('GROQ_API_KEY')
                if groq_key:
                    self.groq_client = Groq(api_key=groq_key)
                    logger.info("✅ Groq client initialized")
                else:
                    logger.warning("⚠️ GROQ_API_KEY not found")
            except Exception as e:
                logger.error(f"Groq initialization error: {e}")
        
        if TAVILY_AVAILABLE:
            try:
                tavily_key = os.getenv('TAVILY_API_KEY')
                if tavily_key:
                    self.tavily_client = TavilyClient(api_key=tavily_key)
                    logger.info("✅ Tavily client initialized")
                else:
                    logger.warning("⚠️ TAVILY_API_KEY not found")
            except Exception as e:
                logger.error(f"Tavily initialization error: {e}")
        
        # Vision models setup
        self.setup_vision_models()
        
        # Camera system
        self.camera = None
        self.vision_mode = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # System prompt
        self.system_prompt = """You are Moi, an advanced AI assistant. You are helpful, friendly, and provide clear responses."""
    
    def setup_vision_models(self):
        """Initialize vision models with graceful fallback"""
        self.vision_available = False
        self.blip_model = None
        self.blip_processor = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Attempting to load vision models...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                model_name = "Salesforce/blip-image-captioning-base"
                self.blip_processor = BlipProcessor.from_pretrained(model_name)
                self.blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
                self.blip_model.to(device)
                self.blip_model.eval()
                
                self.vision_available = True
                logger.info(f"✅ Vision models loaded on {device}")
                
            except Exception as e:
                logger.warning(f"⚠️ Vision models not available: {e}")
                self.vision_available = False
        else:
            logger.info("⚠️ Transformers not available - vision features disabled")
    
    def process_image_vision(self, image: Image.Image) -> Dict[str, Any]:
        """Process image with vision models or fallback"""
        results = {
            'caption': 'Vision processing not available',
            'model_used': 'none',
            'vision_available': self.vision_available
        }
        
        if self.vision_available and self.blip_model and self.blip_processor:
            try:
                # Resize if too large
                if image.width > 512 or image.height > 512:
                    image = image.copy()
                    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                
                inputs = self.blip_processor(image, return_tensors="pt")
                
                with torch.no_grad():
                    out = self.blip_model.generate(**inputs, max_length=50, num_beams=4)
                
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                
                results['caption'] = caption
                results['model_used'] = 'blip'
                
            except Exception as e:
                logger.error(f"Vision processing error: {e}")
                results['caption'] = f"Vision processing failed: {e}"
        
        return results
    
    def chat_with_groq(self, messages: List[Dict[str, str]], model: str = "llama-3.1-8b-instant") -> str:
        """Chat with Groq API with fallback"""
        if not self.groq_client:
            return "Chat functionality not available. Please set GROQ_API_KEY."
        
        try:
            system_message = {"role": "system", "content": self.system_prompt}
            full_messages = [system_message] + messages
            
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=full_messages,
                max_tokens=2048,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Groq chat error: {e}")
            return f"Chat error: {e}"
    
    def web_search(self, query: str) -> List[Dict[str, Any]]:
        """Web search with fallback"""
        if not self.tavily_client:
            return []
        
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="basic",
                max_results=3
            )
            return response.get('results', [])
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    def process_file(self, file_content: bytes, file_type: str) -> Dict[str, Any]:
        """Process uploaded files"""
        try:
            if file_type.startswith('image/'):
                image = self.image_processor.read_image_from_bytes(file_content)
                if image:
                    vision_results = self.process_image_vision(image)
                    return {
                        'type': 'image',
                        'caption': vision_results['caption'],
                        'model_used': vision_results['model_used'],
                        'vision_available': vision_results['vision_available']
                    }
            
            elif file_type == 'text/plain':
                text_content = file_content.decode('utf-8')
                return {
                    'type': 'text',
                    'content': text_content[:1000]  # Limit to first 1000 chars
                }
            
            return {'type': 'unsupported', 'message': 'File type not supported'}
                
        except Exception as e:
            return {'type': 'error', 'message': str(e)}

# Initialize AI Assistant
ai_assistant = HeadlessAIAssistant()

# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        chat_history = data.get('history', [])
        web_search_enabled = data.get('web_search_enabled', False)
        
        # Build messages
        messages = []
        for msg in chat_history[-10:]:  # Last 10 messages
            messages.append({"role": msg['role'], "content": msg['content']})
        
        # Add web search if enabled
        if web_search_enabled and any(keyword in user_message.lower() for keyword in ['search', 'find', 'what is']):
            search_results = ai_assistant.web_search(user_message)
            if search_results:
                context = "\n".join([f"- {result.get('title', '')}: {result.get('content', '')[:200]}" 
                                   for result in search_results[:2]])
                user_message += f"\n\nSearch context:\n{context}"
        
        messages.append({"role": "user", "content": user_message})
        
        response = ai_assistant.chat_with_groq(messages)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        
        if ai_assistant.camera is None:
            ai_assistant.camera = HeadlessCamera()
            
        success = ai_assistant.camera.open()
        
        if success:
            return jsonify({'status': 'Vision mode started (headless simulation)', 'camera_working': True})
        else:
            return jsonify({'error': 'Failed to start camera simulation'}), 400
            
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
                    break
                
                with ai_assistant.frame_lock:
                    ai_assistant.current_frame = frame.copy() if frame is not None else None
                
                if frame is not None:
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(frame)
                    
                    # Convert to bytes
                    frame_bytes = ai_assistant.image_processor.image_to_bytes(pil_image, 'JPEG', 70)
                    
                    if frame_bytes:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.1)  # Control frame rate
                
            except Exception as e:
                logger.error(f"Video feed error: {e}")
                break
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    try:
        with ai_assistant.frame_lock:
            if ai_assistant.current_frame is None:
                return jsonify({'error': 'No frame available'}), 400
            
            frame = ai_assistant.current_frame.copy()
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Process with vision
        vision_results = ai_assistant.process_image_vision(pil_image)
        
        return jsonify({
            'caption': vision_results['caption'],
            'model_used': vision_results['model_used'],
            'vision_available': vision_results['vision_available']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        size = data.get('size', '512x512')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        image_base64 = ai_assistant.image_generator.generate_and_encode(prompt, size)
        
        if image_base64:
            return jsonify({
                'success': True,
                'image': image_base64,
                'prompt': prompt
            })
        else:
            return jsonify({'error': 'Image generation failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge/objects', methods=['GET'])
def get_knowledge_objects():
    try:
        query = request.args.get('query', '')
        limit = int(request.args.get('limit', 20))
        
        objects = ai_assistant.knowledge_db.search_objects(query, limit)
        return jsonify({'objects': objects, 'count': len(objects)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/system-status', methods=['GET'])
def system_status():
    """Get system status and available features"""
    return jsonify({
        'vision_available': ai_assistant.vision_available,
        'groq_available': ai_assistant.groq_client is not None,
        'tavily_available': ai_assistant.tavily_client is not None,
        'transformers_available': TRANSFORMERS_AVAILABLE,
        'pil_available': PIL_AVAILABLE,
        'headless_mode': True,
        'opencv_removed': True
    })

if __name__ == '__main__':
    print("🚀 Starting Moi AI Assistant (Headless Version)...")
    print("✅ OpenCV issues fixed - Pure PIL implementation")
    print("✅ Headless compatible - No GUI dependencies")
    print("🌐 Web interface: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
