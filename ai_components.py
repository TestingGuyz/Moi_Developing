"""
🤖 MOI AI COMPONENTS - PART 2: AI PROCESSING & SECURITY

Advanced AI processing with security hardening and multi-platform support
"""

import os
import cv2
import numpy as np
import hashlib
import hmac
import secrets
import re
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache, wraps
import time
import json
from pathlib import Path

# Continue from app_production.py imports...

# ==================== SECURITY MANAGER ====================

class SecurityManager:
    """Enterprise-grade security manager"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.failed_attempts = defaultdict(list)
        self.blocked_ips = set()
        self.api_keys = self._load_api_keys()
        self.csrf_tokens = {}
        self.rate_limits = defaultdict(list)
        
    def _load_api_keys(self) -> Dict[str, str]:
        """Securely load API keys"""
        return {
            'groq': os.environ.get('GROQ_API_KEY', '').strip(),
            'tavily': os.environ.get('TAVILY_API_KEY', '').strip(),
            'nvidia': os.environ.get('NVIDIA_API_KEY', '').strip(),
            'huggingface': os.environ.get('HUGGINGFACE_TOKEN', '').strip()
        }
    
    def validate_input(self, input_data: Any, input_type: str = 'text') -> Tuple[bool, str]:
        """Comprehensive input validation"""
        try:
            if input_type == 'text':
                return self._validate_text_input(input_data)
            elif input_type == 'file':
                return self._validate_file_input(input_data)
            elif input_type == 'json':
                return self._validate_json_input(input_data)
            else:
                return False, f"Unknown input type: {input_type}"
        except Exception as e:
            logging.error(f"Input validation error: {e}")
            return False, "Input validation failed"
    
    def _validate_text_input(self, text: str) -> Tuple[bool, str]:
        """Validate text input for security issues"""
        if not isinstance(text, str):
            return False, "Input must be a string"
        
        # Length check
        if len(text) > 10000:
            return False, "Input text too long (max 10,000 characters)"
        
        # XSS prevention
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'eval\s*\(',
            r'Function\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return False, f"Potentially dangerous content detected"
        
        # SQL injection prevention
        sql_patterns = [
            r'(union|select|insert|update|delete|drop|create|alter)\s+',
            r'--|#|/\*|\*/',
            r'\bor\b.*\b1\s*=\s*1\b',
            r'\band\b.*\b1\s*=\s*1\b'
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "Potentially dangerous SQL content detected"
        
        return True, "Valid input"
    
    def _validate_file_input(self, file_data: Dict) -> Tuple[bool, str]:
        """Validate file upload for security"""
        if not isinstance(file_data, dict):
            return False, "Invalid file data format"
        
        # Required fields
        required_fields = ['filename', 'content_type', 'size']
        for field in required_fields:
            if field not in file_data:
                return False, f"Missing required field: {field}"
        
        filename = file_data['filename']
        content_type = file_data['content_type']
        file_size = file_data['size']
        
        # Filename validation
        if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
            return False, "Invalid filename characters"
        
        # File size check
        max_size = self.config.max_file_size_mb * 1024 * 1024
        if file_size > max_size:
            return False, f"File too large (max {self.config.max_file_size_mb}MB)"
        
        # Content type validation
        allowed_types = [
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            'text/plain', 'application/json'
        ]
        if content_type not in allowed_types:
            return False, f"File type not allowed: {content_type}"
        
        # Extension validation
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.txt', '.json']
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return False, f"File extension not allowed: {file_ext}"
        
        return True, "Valid file"
    
    def _validate_json_input(self, json_data: Any) -> Tuple[bool, str]:
        """Validate JSON input"""
        try:
            if isinstance(json_data, str):
                json.loads(json_data)
            elif isinstance(json_data, dict):
                json.dumps(json_data)
            else:
                return False, "Invalid JSON data type"
            return True, "Valid JSON"
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
    
    def check_rate_limit(self, identifier: str, max_requests: int = 60, window_minutes: int = 1) -> Tuple[bool, str]:
        """Check rate limiting"""
        current_time = time.time()
        window_seconds = window_minutes * 60
        
        # Clean old requests
        self.rate_limits[identifier] = [
            timestamp for timestamp in self.rate_limits[identifier]
            if current_time - timestamp < window_seconds
        ]
        
        # Check current rate
        current_requests = len(self.rate_limits[identifier])
        if current_requests >= max_requests:
            return False, f"Rate limit exceeded: {current_requests}/{max_requests} requests per {window_minutes} minute(s)"
        
        # Add current request
        self.rate_limits[identifier].append(current_time)
        return True, "Rate limit OK"
    
    def sanitize_output(self, output: str) -> str:
        """Sanitize output for safe display"""
        if not isinstance(output, str):
            return str(output)
        
        # HTML escape
        html_escape_table = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '/': '&#x2F;'
        }
        
        sanitized = output
        for char, escape in html_escape_table.items():
            sanitized = sanitized.replace(char, escape)
        
        return sanitized
    
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token"""
        token = secrets.token_urlsafe(32)
        self.csrf_tokens[session_id] = {
            'token': token,
            'created': time.time(),
            'used': False
        }
        return token
    
    def validate_csrf_token(self, session_id: str, token: str) -> bool:
        """Validate CSRF token"""
        if session_id not in self.csrf_tokens:
            return False
        
        token_data = self.csrf_tokens[session_id]
        
        # Check expiration (1 hour)
        if time.time() - token_data['created'] > 3600:
            del self.csrf_tokens[session_id]
            return False
        
        # Check token match
        if not hmac.compare_digest(token_data['token'], token):
            return False
        
        # Mark as used (one-time use)
        token_data['used'] = True
        return True
    
    def log_security_event(self, event_type: str, details: Dict, severity: str = 'INFO'):
        """Log security events"""
        security_logger = logging.getLogger('security')
        
        event_data = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details
        }
        
        log_message = f"{severity}: {event_type} | {json.dumps(event_data)}"
        
        if severity == 'CRITICAL':
            security_logger.critical(log_message)
        elif severity == 'WARNING':
            security_logger.warning(log_message)
        else:
            security_logger.info(log_message)

# ==================== ADVANCED IMAGE PROCESSOR ====================

class AdvancedImageProcessor:
    """Production-grade image processing with comprehensive error handling"""
    
    def __init__(self, config: SystemConfig, security_manager: SecurityManager):
        self.config = config
        self.security = security_manager
        self.supported_formats = ['JPEG', 'PNG', 'WebP', 'BMP', 'TIFF']
        self.processing_cache = {}
        self.cache_lock = threading.Lock()
        
    def process_image_bytes(self, image_bytes: bytes, operations: List[str] = None) -> Optional[Image.Image]:
        """Process image from bytes with caching and error handling"""
        if operations is None:
            operations = ['validate', 'optimize']
            
        try:
            # Generate cache key
            cache_key = hashlib.md5(image_bytes + str(operations).encode()).hexdigest()
            
            # Check cache
            with self.cache_lock:
                if cache_key in self.processing_cache:
                    return self.processing_cache[cache_key].copy()
            
            # Process image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Apply operations
            for operation in operations:
                image = self._apply_operation(image, operation)
                if image is None:
                    return None
            
            # Cache result
            with self.cache_lock:
                if len(self.processing_cache) > 100:  # Limit cache size
                    oldest_key = next(iter(self.processing_cache))
                    del self.processing_cache[oldest_key]
                self.processing_cache[cache_key] = image.copy()
            
            return image
            
        except Exception as e:
            logging.error(f"Image processing failed: {e}")
            return None
    
    def _apply_operation(self, image: Image.Image, operation: str) -> Optional[Image.Image]:
        """Apply specific image operation"""
        try:
            if operation == 'validate':
                return self._validate_image(image)
            elif operation == 'optimize':
                return self._optimize_image(image)
            elif operation == 'resize':
                return self._resize_image(image)
            elif operation == 'enhance':
                return self._enhance_image(image)
            elif operation == 'normalize':
                return self._normalize_image(image)
            else:
                logging.warning(f"Unknown image operation: {operation}")
                return image
        except Exception as e:
            logging.error(f"Image operation '{operation}' failed: {e}")
            return None
    
    def _validate_image(self, image: Image.Image) -> Optional[Image.Image]:
        """Validate image for security and format issues"""
        # Check dimensions
        if image.width > self.config.max_image_dimension or image.height > self.config.max_image_dimension:
            logging.warning(f"Image too large: {image.width}x{image.height}")
            # Resize to maximum allowed size
            image.thumbnail((self.config.max_image_dimension, self.config.max_image_dimension), Image.Resampling.LANCZOS)
        
        # Check format
        if image.format not in self.supported_formats:
            logging.warning(f"Unsupported image format: {image.format}")
            # Convert to supported format
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            image.format = 'JPEG'
        
        return image
    
    def _optimize_image(self, image: Image.Image) -> Image.Image:
        """Optimize image for performance"""
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[-1])
            else:
                background.paste(image)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def _resize_image(self, image: Image.Image, target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """Resize image maintaining aspect ratio"""
        image_copy = image.copy()
        image_copy.thumbnail(target_size, Image.Resampling.LANCZOS)
        return image_copy
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance image quality"""
        # Auto-enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # Auto-enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    def _normalize_image(self, image: Image.Image) -> Image.Image:
        """Normalize image for ML processing"""
        # Ensure standard size for ML models
        if image.width != 224 or image.height != 224:
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        return image
    
    def image_to_base64(self, image: Image.Image, format: str = 'JPEG', quality: int = 85) -> Optional[str]:
        """Convert image to base64 with optimization"""
        try:
            buffer = io.BytesIO()
            
            # Optimize based on format
            save_kwargs = {}
            if format.upper() == 'JPEG':
                save_kwargs = {
                    'quality': quality,
                    'optimize': True,
                    'progressive': True
                }
            elif format.upper() == 'PNG':
                save_kwargs = {
                    'optimize': True,
                    'compress_level': 6
                }
            
            image.save(buffer, format=format, **save_kwargs)
            image_bytes = buffer.getvalue()
            
            return base64.b64encode(image_bytes).decode('utf-8')
            
        except Exception as e:
            logging.error(f"Base64 encoding failed: {e}")
            return None
    
    def detect_image_properties(self, image: Image.Image) -> Dict[str, Any]:
        """Detect comprehensive image properties"""
        try:
            # Basic properties
            properties = {
                'width': image.width,
                'height': image.height,
                'mode': image.mode,
                'format': image.format,
                'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
            
            # Calculate file size estimate
            buffer = io.BytesIO()
            image.save(buffer, format=image.format or 'JPEG')
            properties['estimated_size_bytes'] = len(buffer.getvalue())
            
            # Color analysis
            if image.mode in ('RGB', 'RGBA'):
                # Convert to RGB for analysis
                rgb_image = image.convert('RGB')
                
                # Get dominant colors
                colors = rgb_image.getcolors(maxcolors=256*256*256)
                if colors:
                    # Sort by frequency
                    colors.sort(reverse=True)
                    properties['dominant_colors'] = [color[1] for color in colors[:5]]
                
                # Calculate average brightness
                grayscale = rgb_image.convert('L')
                stat = ImageStat.Stat(grayscale)
                properties['average_brightness'] = stat.mean[0]
                properties['brightness_stddev'] = stat.stddev[0]
            
            return properties
            
        except Exception as e:
            logging.error(f"Image analysis failed: {e}")
            return {'error': str(e)}
    
    def create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (128, 128)) -> Optional[Image.Image]:
        """Create optimized thumbnail"""
        try:
            # Create thumbnail maintaining aspect ratio
            thumbnail = image.copy()
            thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Add padding if needed to match exact size
            if thumbnail.size != size:
                padded = Image.new('RGB', size, (255, 255, 255))
                paste_x = (size[0] - thumbnail.width) // 2
                paste_y = (size[1] - thumbnail.height) // 2
                padded.paste(thumbnail, (paste_x, paste_y))
                thumbnail = padded
            
            return thumbnail
            
        except Exception as e:
            logging.error(f"Thumbnail creation failed: {e}")
            return None

# ==================== INTELLIGENT MODEL MANAGER ====================

class IntelligentModelManager:
    """Advanced model management with dynamic loading and optimization"""
    
    def __init__(self, config: SystemConfig, resource_monitor: ResourceMonitor):
        self.config = config
        self.monitor = resource_monitor
        self.models = {}
        self.model_stats = defaultdict(dict)
        self.loading_locks = defaultdict(threading.Lock)
        self.model_queue = []
        self.max_models_in_memory = 3
        
        # Device detection
        self.device = self._detect_optimal_device()
        logging.info(f"💻 Optimal device detected: {self.device}")
        
        # Model configurations
        self.model_configs = {
            'vision': {
                'name': 'Salesforce/blip-image-captioning-base',
                'type': 'vision',
                'memory_mb': 1024,
                'priority': 1
            },
            'text': {
                'name': 'sentence-transformers/all-MiniLM-L6-v2',
                'type': 'text',
                'memory_mb': 512,
                'priority': 2
            }
        }
    
    def _detect_optimal_device(self) -> str:
        """Detect optimal computation device"""
        try:
            if TORCH_AVAILABLE:
                # Check for CUDA
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logging.info(f"🚀 CUDA GPU detected: {device_name} ({memory_gb:.1f}GB)")
                    return 'cuda'
                
                # Check for MPS (Apple Silicon)
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    logging.info("🍎 Apple Silicon GPU (MPS) detected")
                    return 'mps'
            
            # Fallback to CPU
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / 1024**3
            logging.info(f"💻 Using CPU: {cpu_count} cores, {memory_gb:.1f}GB RAM")
            return 'cpu'
            
        except Exception as e:
            logging.warning(f"Device detection error: {e}, defaulting to CPU")
            return 'cpu'
    
    def load_model(self, model_key: str, force_reload: bool = False) -> Optional[Any]:
        """Load model with intelligent memory management"""
        if model_key not in self.model_configs:
            logging.error(f"Unknown model key: {model_key}")
            return None
        
        # Check if model already loaded
        if model_key in self.models and not force_reload:
            self._update_model_usage(model_key)
            return self.models[model_key]
        
        # Thread-safe loading
        with self.loading_locks[model_key]:
            # Double-check after acquiring lock
            if model_key in self.models and not force_reload:
                self._update_model_usage(model_key)
                return self.models[model_key]
            
            return self._load_model_internal(model_key)
    
    def _load_model_internal(self, model_key: str) -> Optional[Any]:
        """Internal model loading with error handling"""
        config = self.model_configs[model_key]
        start_time = time.time()
        
        try:
            # Check available memory
            current_memory = self.monitor.get_current_metrics()['process']['memory_rss_mb']
            required_memory = config['memory_mb']
            
            if current_memory + required_memory > self.config.max_model_memory_mb:
                self._free_least_used_model()
            
            logging.info(f"📈 Loading model: {config['name']}")
            
            # Load based on model type
            if config['type'] == 'vision':
                model = self._load_vision_model(config)
            elif config['type'] == 'text':
                model = self._load_text_model(config)
            else:
                raise ValueError(f"Unknown model type: {config['type']}")
            
            if model is None:
                return None
            
            # Store model and update stats
            self.models[model_key] = model
            self.model_stats[model_key] = {
                'loaded_at': time.time(),
                'load_time': time.time() - start_time,
                'usage_count': 0,
                'last_used': time.time(),
                'memory_mb': required_memory
            }
            
            # Update model queue for LRU
            if model_key in self.model_queue:
                self.model_queue.remove(model_key)
            self.model_queue.append(model_key)
            
            logging.info(f"✅ Model loaded: {model_key} ({time.time() - start_time:.2f}s)")
            return model
            
        except Exception as e:
            logging.error(f"❌ Model loading failed: {model_key} | {e}")
            return None
    
    def _load_vision_model(self, config: Dict) -> Optional[Any]:
        """Load vision model with comprehensive error handling"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                logging.error("Transformers not available for vision model")
                return None
            
            model_name = config['name']
            
            # Load processor and model
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name)
            
            # Move to optimal device
            if self.device != 'cpu':
                model = model.to(self.device)
            
            # Set to evaluation mode
            model.eval()
            
            # Enable optimizations
            if hasattr(model, 'half') and self.device == 'cuda':
                model = model.half()  # Use half precision for memory efficiency
            
            return {
                'processor': processor,
                'model': model,
                'type': 'vision',
                'device': self.device
            }
            
        except Exception as e:
            logging.error(f"Vision model loading failed: {e}")
            return None
    
    def _load_text_model(self, config: Dict) -> Optional[Any]:
        """Load text model with error handling"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                logging.error("Transformers not available for text model")
                return None
            
            model_name = config['name']
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Move to optimal device
            if self.device != 'cpu':
                model = model.to(self.device)
            
            model.eval()
            
            return {
                'tokenizer': tokenizer,
                'model': model,
                'type': 'text',
                'device': self.device
            }
            
        except Exception as e:
            logging.error(f"Text model loading failed: {e}")
            return None
    
    def _update_model_usage(self, model_key: str):
        """Update model usage statistics"""
        if model_key in self.model_stats:
            self.model_stats[model_key]['usage_count'] += 1
            self.model_stats[model_key]['last_used'] = time.time()
            
            # Update LRU queue
            if model_key in self.model_queue:
                self.model_queue.remove(model_key)
            self.model_queue.append(model_key)
    
    def _free_least_used_model(self):
        """Free the least recently used model"""
        if not self.model_queue:
            return
        
        # Find least recently used model
        lru_model = self.model_queue[0]
        
        logging.info(f"🗑️ Freeing model to save memory: {lru_model}")
        
        # Remove model
        if lru_model in self.models:
            del self.models[lru_model]
        
        if lru_model in self.model_stats:
            del self.model_stats[lru_model]
        
        self.model_queue.remove(lru_model)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def process_image_with_vision(self, image: Image.Image, prompt: str = "") -> Dict[str, Any]:
        """Process image with vision model"""
        model = self.load_model('vision')
        if model is None:
            return {
                'error': 'Vision model not available',
                'caption': 'Vision processing unavailable'
            }
        
        try:
            start_time = time.time()
            
            # Prepare inputs
            if prompt:
                inputs = model['processor'](image, prompt, return_tensors="pt")
            else:
                inputs = model['processor'](image, return_tensors="pt")
            
            # Move inputs to device
            if model['device'] != 'cpu':
                inputs = {k: v.to(model['device']) for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                out = model['model'].generate(**inputs, max_length=50, num_beams=4)
            
            # Decode output
            caption = model['processor'].decode(out[0], skip_special_tokens=True)
            
            processing_time = time.time() - start_time
            
            return {
                'caption': caption,
                'processing_time': processing_time,
                'model_used': 'blip',
                'device': model['device'],
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Vision processing error: {e}")
            return {
                'error': str(e),
                'caption': 'Vision processing failed',
                'success': False
            }
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics"""
        return {
            'loaded_models': list(self.models.keys()),
            'model_stats': dict(self.model_stats),
            'device': self.device,
            'model_queue': list(self.model_queue),
            'available_models': list(self.model_configs.keys())
        }
    
    def cleanup_models(self):
        """Clean up all models"""
        logging.info("🗑️ Cleaning up all models")
        
        self.models.clear()
        self.model_stats.clear()
        self.model_queue.clear()
        
        import gc
        gc.collect()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

# Continue with the Flask app and routes in the next part...
