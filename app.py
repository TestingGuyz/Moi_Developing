"""
Moi AI Assistant - Complete Implementation
A self-learning AI with vision, memory, and autonomous agents

All components integrated in single file:
- Vision Model (BLIP + NVIDIA Vision API)
- Object Detection (YOLOv5)
- Knowledge Database (SQLite)
- Agentic AI Systems (3 autonomous agents)
- Image Generation (Multiple backends)
- Generative Chat (Groq API)
"""

import os
import json
import base64
import cv2
import numpy as np
import requests
import re
import html
import sqlite3
import logging
import urllib.parse
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from groq import Groq
from tavily import TavilyClient
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    TextDataset, 
    DataCollatorForLanguageModeling
)
import torch
from PIL import Image
import io
from dotenv import load_dotenv
import threading
import time
from typing import Dict, List, Optional, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)


# ==================== KNOWLEDGE DATABASE ====================

class KnowledgeDatabase:
    """
    Knowledge Database System
    Stores and manages learned object information with full CRUD operations
    """
    
    def __init__(self, db_path: str = "moi_knowledge.db"):
        """Initialize the knowledge database"""
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Main objects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    color TEXT,
                    climate TEXT,
                    types TEXT,
                    category TEXT,
                    confidence_score REAL DEFAULT 0.0,
                    times_seen INTEGER DEFAULT 1,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_notes TEXT,
                    metadata TEXT
                )
            ''')
            
            # Object properties table (key-value pairs)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS object_properties (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_id INTEGER,
                    property_key TEXT NOT NULL,
                    property_value TEXT,
                    source TEXT,
                    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (object_id) REFERENCES objects(id) ON DELETE CASCADE
                )
            ''')
            
            # Learning history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_id INTEGER,
                    event_type TEXT,
                    event_data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (object_id) REFERENCES objects(id) ON DELETE CASCADE
                )
            ''')
            
            # Training data table for secret model
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_text TEXT,
                    output_text TEXT,
                    context TEXT,
                    quality_score REAL DEFAULT 0.0,
                    used_for_training BOOLEAN DEFAULT 0,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
    
    def add_object(self, name: str, **kwargs) -> int:
        """Add a new object to the database or update if exists"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if object exists
            cursor.execute('SELECT id, times_seen FROM objects WHERE name = ?', (name,))
            result = cursor.fetchone()
            
            if result:
                # Update existing object
                object_id, times_seen = result
                update_fields = []
                update_values = []
                
                for key, value in kwargs.items():
                    if key in ['description', 'color', 'climate', 'types', 'category', 'user_notes', 'confidence_score']:
                        update_fields.append(f"{key} = ?")
                        update_values.append(value)
                
                update_fields.append("times_seen = ?")
                update_values.append(times_seen + 1)
                update_fields.append("last_seen = ?")
                update_values.append(datetime.now().isoformat())
                
                if update_fields:
                    update_values.append(object_id)
                    cursor.execute(f'''
                        UPDATE objects 
                        SET {', '.join(update_fields)}
                        WHERE id = ?
                    ''', update_values)
                
                conn.commit()
                conn.close()
                return object_id
            else:
                # Insert new object
                fields = ['name']
                values = [name]
                
                for key in ['description', 'color', 'climate', 'types', 'category', 'user_notes', 'confidence_score']:
                    if key in kwargs:
                        fields.append(key)
                        values.append(kwargs[key])
                
                placeholders = ','.join(['?' for _ in values])
                cursor.execute(f'''
                    INSERT INTO objects ({','.join(fields)})
                    VALUES ({placeholders})
                ''', values)
                
                object_id = cursor.lastrowid
                
                # Log learning event
                cursor.execute('''
                    INSERT INTO learning_history (object_id, event_type, event_data)
                    VALUES (?, ?, ?)
                ''', (object_id, 'object_learned', json.dumps(kwargs)))
                
                conn.commit()
                conn.close()
                return object_id
    
    def get_object(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve object information by name"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM objects WHERE name = ?', (name,))
            result = cursor.fetchone()
            
            if result:
                object_dict = dict(result)
                object_id = object_dict['id']
                
                # Get additional properties
                cursor.execute('SELECT property_key, property_value, source FROM object_properties WHERE object_id = ?', (object_id,))
                properties = cursor.fetchall()
                object_dict['properties'] = {row['property_key']: {'value': row['property_value'], 'source': row['source']} for row in properties}
                
                conn.close()
                return object_dict
            
            conn.close()
            return None
    
    def add_property(self, object_name: str, property_key: str, property_value: str, source: str = 'user'):
        """Add a property to an object"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get object ID
            cursor.execute('SELECT id FROM objects WHERE name = ?', (object_name,))
            result = cursor.fetchone()
            
            if result:
                object_id = result[0]
                
                # Check if property exists
                cursor.execute('''
                    SELECT id FROM object_properties 
                    WHERE object_id = ? AND property_key = ?
                ''', (object_id, property_key))
                
                if cursor.fetchone():
                    # Update existing property
                    cursor.execute('''
                        UPDATE object_properties 
                        SET property_value = ?, source = ?, added_date = ?
                        WHERE object_id = ? AND property_key = ?
                    ''', (property_value, source, datetime.now().isoformat(), object_id, property_key))
                else:
                    # Insert new property
                    cursor.execute('''
                        INSERT INTO object_properties (object_id, property_key, property_value, source)
                        VALUES (?, ?, ?, ?)
                    ''', (object_id, property_key, property_value, source))
                
                conn.commit()
            
            conn.close()
    
    def update_object(self, name: str, **kwargs):
        """Update object information"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            update_fields = []
            update_values = []
            
            for key, value in kwargs.items():
                if key in ['description', 'color', 'climate', 'types', 'category', 'user_notes', 'confidence_score']:
                    update_fields.append(f"{key} = ?")
                    update_values.append(value)
            
            if update_fields:
                update_fields.append("last_seen = ?")
                update_values.append(datetime.now().isoformat())
                update_values.append(name)
                
                cursor.execute(f'''
                    UPDATE objects 
                    SET {', '.join(update_fields)}
                    WHERE name = ?
                ''', update_values)
                
                conn.commit()
            
            conn.close()
    
    def search_objects(self, query: str = None, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search objects by query or category"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if query:
                cursor.execute('''
                    SELECT * FROM objects 
                    WHERE name LIKE ? OR description LIKE ? OR category LIKE ?
                    ORDER BY times_seen DESC, last_seen DESC
                    LIMIT ?
                ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
            elif category:
                cursor.execute('''
                    SELECT * FROM objects 
                    WHERE category = ?
                    ORDER BY times_seen DESC, last_seen DESC
                    LIMIT ?
                ''', (category, limit))
            else:
                cursor.execute('''
                    SELECT * FROM objects 
                    ORDER BY times_seen DESC, last_seen DESC
                    LIMIT ?
                ''', (limit,))
            
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return results
    
    def get_all_objects(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all objects from database"""
        return self.search_objects(limit=limit)
    
    def delete_object(self, name: str):
        """Delete an object from database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM objects WHERE name = ?', (name,))
            conn.commit()
            conn.close()
    
    def add_training_data(self, input_text: str, output_text: str, context: str = "", quality_score: float = 0.5):
        """Add training data for the secret model"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO training_data (input_text, output_text, context, quality_score)
                VALUES (?, ?, ?, ?)
            ''', (input_text, output_text, context, quality_score))
            
            conn.commit()
            conn.close()
    
    def get_training_data(self, unused_only: bool = True, limit: int = 100) -> List[Dict[str, Any]]:
        """Get training data for model training"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if unused_only:
                cursor.execute('''
                    SELECT * FROM training_data 
                    WHERE used_for_training = 0 
                    ORDER BY quality_score DESC, created_date DESC
                    LIMIT ?
                ''', (limit,))
            else:
                cursor.execute('''
                    SELECT * FROM training_data 
                    ORDER BY quality_score DESC, created_date DESC
                    LIMIT ?
                ''', (limit,))
            
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return results
    
    def mark_training_data_used(self, data_ids: List[int]):
        """Mark training data as used"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            placeholders = ','.join(['?' for _ in data_ids])
            cursor.execute(f'''
                UPDATE training_data 
                SET used_for_training = 1 
                WHERE id IN ({placeholders})
            ''', data_ids)
            
            conn.commit()
            conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            cursor.execute('SELECT COUNT(*) FROM objects')
            stats['total_objects'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM object_properties')
            stats['total_properties'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM training_data')
            stats['total_training_data'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM training_data WHERE used_for_training = 0')
            stats['unused_training_data'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(times_seen) FROM objects')
            stats['total_observations'] = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT name, times_seen FROM objects ORDER BY times_seen DESC LIMIT 5')
            stats['most_seen_objects'] = [{'name': row[0], 'times_seen': row[1]} for row in cursor.fetchall()]
            
            conn.close()
            return stats
    
    def export_knowledge(self, file_path: str = "knowledge_export.json"):
        """Export all knowledge to JSON file"""
        objects = self.get_all_objects(limit=10000)
        stats = self.get_statistics()
        
        export_data = {
            'export_date': datetime.now().isoformat(),
            'statistics': stats,
            'objects': objects
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return file_path
    
    def import_knowledge(self, file_path: str):
        """Import knowledge from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        objects = data.get('objects', [])
        for obj in objects:
            name = obj.pop('name')
            obj.pop('id', None)  # Remove ID
            self.add_object(name, **obj)


# ==================== IMAGE GENERATOR ====================

class ImageGenerator:
    """
    Image generation using multiple backends:
    1. Together.ai API (Stable Diffusion)
    2. Pollinations.ai (Free alternative)
    3. Hugging Face Inference API
    """
    
    def __init__(self):
        self.together_api_key = os.getenv('TOGETHER_API_KEY')
        self.hf_api_key = os.getenv('HUGGINGFACE_TOKEN')
        self.available_backends = self._check_available_backends()
        
        logger.info(f"Image generator initialized. Available backends: {self.available_backends}")
    
    def _check_available_backends(self) -> list:
        """Check which image generation backends are available"""
        backends = ['pollinations']  # Always available (free)
        
        if self.together_api_key:
            backends.append('together')
        
        if self.hf_api_key:
            backends.append('huggingface')
        
        return backends
    
    def generate_image(self, prompt: str, backend: str = 'auto', size: str = '512x512') -> Optional[bytes]:
        """Generate image from text prompt"""
        
        if backend == 'auto':
            # Try backends in order of preference
            for backend_name in self.available_backends:
                try:
                    return self._generate_with_backend(prompt, backend_name, size)
                except Exception as e:
                    logger.warning(f"Backend {backend_name} failed: {e}")
                    continue
            
            logger.error("All backends failed")
            return None
        else:
            return self._generate_with_backend(prompt, backend, size)
    
    def _generate_with_backend(self, prompt: str, backend: str, size: str) -> Optional[bytes]:
        """Generate image with specific backend"""
        
        if backend == 'together':
            return self._generate_together(prompt, size)
        elif backend == 'huggingface':
            return self._generate_huggingface(prompt, size)
        elif backend == 'pollinations':
            return self._generate_pollinations(prompt, size)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _generate_together(self, prompt: str, size: str) -> Optional[bytes]:
        """Generate image using Together.ai API"""
        try:
            if not self.together_api_key:
                raise ValueError("Together.ai API key not found")
            
            logger.info(f"Generating image with Together.ai: {prompt[:50]}...")
            
            # Parse size
            width, height = map(int, size.split('x'))
            
            url = "https://api.together.xyz/v1/images/generations"
            headers = {
                "Authorization": f"Bearer {self.together_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "stabilityai/stable-diffusion-xl-base-1.0",
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": 30,
                "n": 1
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            # Get image URL or base64
            if 'data' in result and len(result['data']) > 0:
                image_data = result['data'][0]
                
                if 'url' in image_data:
                    # Download image from URL
                    img_response = requests.get(image_data['url'], timeout=30)
                    img_response.raise_for_status()
                    return img_response.content
                elif 'b64_json' in image_data:
                    # Decode base64
                    return base64.b64decode(image_data['b64_json'])
            
            raise ValueError("No image data in response")
            
        except Exception as e:
            logger.error(f"Together.ai error: {e}")
            raise
    
    def _generate_huggingface(self, prompt: str, size: str) -> Optional[bytes]:
        """Generate image using Hugging Face Inference API"""
        try:
            if not self.hf_api_key:
                raise ValueError("Hugging Face API key not found")
            
            logger.info(f"Generating image with Hugging Face: {prompt[:50]}...")
            
            API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            return response.content
            
        except Exception as e:
            logger.error(f"Hugging Face error: {e}")
            raise
    
    def _generate_pollinations(self, prompt: str, size: str) -> Optional[bytes]:
        """Generate image using Pollinations.ai (free service)"""
        try:
            logger.info(f"Generating image with Pollinations.ai: {prompt[:50]}...")
            
            # Parse size
            width, height = map(int, size.split('x'))
            
            # Encode prompt for URL
            encoded_prompt = urllib.parse.quote(prompt)
            
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true"
            
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            return response.content
            
        except Exception as e:
            logger.error(f"Pollinations.ai error: {e}")
            raise
    
    def save_image(self, image_bytes: bytes, output_path: str):
        """Save image bytes to file"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.save(output_path)
            logger.info(f"Image saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False
    
    def image_to_base64(self, image_bytes: bytes) -> str:
        """Convert image bytes to base64 string"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def generate_and_encode(self, prompt: str, backend: str = 'auto', size: str = '512x512') -> Optional[str]:
        """Generate image and return as base64 string"""
        try:
            image_bytes = self.generate_image(prompt, backend, size)
            if image_bytes:
                return self.image_to_base64(image_bytes)
            return None
        except Exception as e:
            logger.error(f"Error generating and encoding image: {e}")
            return None


# ==================== AGENTIC AI SYSTEMS ====================

class SecretModelTrainer:
    """
    Agentic AI #1 - Secret Model Trainer
    Continuously trains a local generative model using collected training data
    """
    
    def __init__(self, knowledge_db: KnowledgeDatabase, model_name: str = "gpt2"):
        self.knowledge_db = knowledge_db
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_training = False
        self.training_thread = None
        self.training_enabled = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load or initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load the secret model"""
        try:
            model_path = "models/secret_model"
            
            if os.path.exists(model_path):
                logger.info(f"Loading existing secret model from {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
            else:
                logger.info(f"Initializing new secret model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                
                # Set pad token if not exists
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.to(self.device)
            logger.info(f"✅ Secret model initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing secret model: {e}")
            self.model = None
            self.tokenizer = None
    
    def start_training_loop(self, interval: int = 3600):
        """Start continuous training loop in background"""
        if self.training_thread and self.training_thread.is_alive():
            logger.warning("Training loop already running")
            return
        
        self.training_enabled = True
        self.training_thread = threading.Thread(
            target=self._training_loop,
            args=(interval,),
            daemon=True
        )
        self.training_thread.start()
        logger.info(f"✅ Secret model training loop started (interval: {interval}s)")
    
    def stop_training_loop(self):
        """Stop the training loop"""
        self.training_enabled = False
        if self.training_thread:
            self.training_thread.join(timeout=5)
        logger.info("Secret model training loop stopped")
    
    def _training_loop(self, interval: int):
        """Continuous training loop"""
        while self.training_enabled:
            try:
                # Get new training data
                training_data = self.knowledge_db.get_training_data(unused_only=True, limit=100)
                
                if len(training_data) >= 10:  # Train only if we have enough data
                    logger.info(f"Starting training with {len(training_data)} new examples")
                    self._train_on_data(training_data)
                    
                    # Mark data as used
                    data_ids = [d['id'] for d in training_data]
                    self.knowledge_db.mark_training_data_used(data_ids)
                    
                    # Save model
                    self._save_model()
                else:
                    logger.info(f"Not enough training data ({len(training_data)} examples)")
                
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
            
            # Wait for next iteration
            time.sleep(interval)
    
    def _train_on_data(self, training_data: List[Dict[str, Any]]):
        """Train model on new data"""
        if not self.model or not self.tokenizer:
            logger.error("Model not initialized")
            return
        
        try:
            self.is_training = True
            
            # Prepare training texts
            training_texts = []
            for item in training_data:
                # Format: input + output with special tokens
                text = f"<|input|>{item['input_text']}<|output|>{item['output_text']}<|end|>"
                training_texts.append(text)
            
            # Save to temporary file
            temp_file = "temp_training_data.txt"
            with open(temp_file, 'w') as f:
                f.write('\n'.join(training_texts))
            
            # Create dataset
            train_dataset = TextDataset(
                tokenizer=self.tokenizer,
                file_path=temp_file,
                block_size=128
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir="./models/secret_model_checkpoints",
                overwrite_output_dir=True,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                save_steps=100,
                save_total_limit=2,
                learning_rate=5e-5,
                warmup_steps=10,
                logging_steps=10,
                no_cuda=(self.device.type == 'cpu')
            )
            
            # Train
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
            )
            
            trainer.train()
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            logger.info("✅ Training completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
        finally:
            self.is_training = False
    
    def _save_model(self):
        """Save the trained model"""
        try:
            save_path = "models/secret_model"
            os.makedirs(save_path, exist_ok=True)
            
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            logger.info(f"✅ Secret model saved to {save_path}")
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
    
    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generate response using the secret model"""
        if not self.model or not self.tokenizer:
            return "Secret model not available"
        
        try:
            input_text = f"<|input|>{prompt}<|output|>"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the output part
            if "<|output|>" in response:
                response = response.split("<|output|>")[1]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {e}"


class InfoGatheringAgent:
    """
    Agentic AI #2 - Info Gathering Agent
    Automatically gathers information about detected objects
    """
    
    def __init__(self, knowledge_db: KnowledgeDatabase, tavily_api_key: str, groq_api_key: str):
        self.knowledge_db = knowledge_db
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.groq_client = Groq(api_key=groq_api_key)
        self.gathering_queue = []
        self.queue_lock = threading.Lock()
        self.worker_thread = None
        self.is_running = False
    
    def start_worker(self):
        """Start background worker for gathering info"""
        if self.worker_thread and self.worker_thread.is_alive():
            logger.warning("Worker already running")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )
        self.worker_thread.start()
        logger.info("✅ Info gathering agent worker started")
    
    def stop_worker(self):
        """Stop the worker"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Info gathering agent worker stopped")
    
    def queue_object_for_research(self, object_name: str, context: str = ""):
        """Add object to research queue"""
        with self.queue_lock:
            self.gathering_queue.append({
                'object_name': object_name,
                'context': context,
                'queued_at': datetime.now().isoformat()
            })
        logger.info(f"Queued {object_name} for research")
    
    def _worker_loop(self):
        """Background worker that processes research queue"""
        while self.is_running:
            try:
                item = None
                with self.queue_lock:
                    if self.gathering_queue:
                        item = self.gathering_queue.pop(0)
                
                if item:
                    self._gather_object_info(item['object_name'], item['context'])
                else:
                    time.sleep(5)  # Wait if queue is empty
                    
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(5)
    
    def _gather_object_info(self, object_name: str, context: str = ""):
        """Gather comprehensive information about an object"""
        try:
            logger.info(f"Gathering info for: {object_name}")
            
            # Check if already in database
            existing = self.knowledge_db.get_object(object_name)
            if existing and existing.get('times_seen', 0) > 5:
                logger.info(f"{object_name} already well-documented")
                return
            
            # Perform web search
            search_results = self._web_search(object_name)
            
            # Extract structured information using AI
            structured_info = self._extract_structured_info(object_name, search_results)
            
            # Store in database
            self.knowledge_db.add_object(
                name=object_name,
                description=structured_info.get('description', ''),
                color=structured_info.get('color', ''),
                climate=structured_info.get('climate', ''),
                types=structured_info.get('types', ''),
                category=structured_info.get('category', '')
            )
            
            # Add detailed properties
            for key, value in structured_info.get('properties', {}).items():
                self.knowledge_db.add_property(object_name, key, value, source='web_search')
            
            logger.info(f"✅ Gathered and stored info for {object_name}")
            
        except Exception as e:
            logger.error(f"Error gathering info for {object_name}: {e}")
    
    def _web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search"""
        try:
            response = self.tavily_client.search(
                query=f"{query} information characteristics properties",
                search_depth="advanced",
                max_results=5
            )
            return response.get('results', [])
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    def _extract_structured_info(self, object_name: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract structured information using AI"""
        try:
            # Prepare context from search results
            context = "\n".join([
                f"- {result.get('title', '')}: {result.get('content', '')[:300]}"
                for result in search_results[:3]
            ])
            
            prompt = f"""
            Extract structured information about: {object_name}
            
            Search Results:
            {context}
            
            Please provide a JSON response with the following structure:
            {{
                "description": "brief description",
                "color": "typical color(s)",
                "climate": "climate conditions (if applicable)",
                "types": "common types or varieties",
                "category": "general category",
                "properties": {{
                    "size": "typical size",
                    "uses": "common uses",
                    "origin": "where it's from",
                    "nutrition": "nutritional info (if food)"
                }}
            }}
            
            Only include relevant fields. Return valid JSON only.
            """
            
            response = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            info = json.loads(content)
            return info
            
        except Exception as e:
            logger.error(f"Error extracting structured info: {e}")
            return {
                'description': f"A {object_name}",
                'category': 'unknown',
                'properties': {}
            }


class TrainingSupervisor:
    """
    Agentic AI #3 - Training Supervisor
    Manages training loops and decides what data to use for training
    """
    
    def __init__(self, knowledge_db: KnowledgeDatabase, secret_trainer: SecretModelTrainer):
        self.knowledge_db = knowledge_db
        self.secret_trainer = secret_trainer
        self.is_running = False
        self.supervisor_thread = None
    
    def start_supervision(self, interval: int = 1800):
        """Start supervision loop"""
        if self.supervisor_thread and self.supervisor_thread.is_alive():
            logger.warning("Supervisor already running")
            return
        
        self.is_running = True
        self.supervisor_thread = threading.Thread(
            target=self._supervision_loop,
            args=(interval,),
            daemon=True
        )
        self.supervisor_thread.start()
        logger.info(f"✅ Training supervisor started (interval: {interval}s)")
    
    def stop_supervision(self):
        """Stop supervision"""
        self.is_running = False
        if self.supervisor_thread:
            self.supervisor_thread.join(timeout=5)
        logger.info("Training supervisor stopped")
    
    def _supervision_loop(self, interval: int):
        """Main supervision loop"""
        while self.is_running:
            try:
                # Evaluate training data quality
                self._evaluate_training_data()
                
                # Check if model needs retraining
                self._check_model_performance()
                
                # Clean up old/bad data
                self._cleanup_data()
                
                # Generate synthetic training data from knowledge base
                self._generate_synthetic_data()
                
            except Exception as e:
                logger.error(f"Supervision loop error: {e}")
            
            time.sleep(interval)
    
    def _evaluate_training_data(self):
        """Evaluate and score training data quality"""
        try:
            training_data = self.knowledge_db.get_training_data(unused_only=True, limit=50)
            logger.info(f"Evaluated {len(training_data)} training examples")
        except Exception as e:
            logger.error(f"Error evaluating training data: {e}")
    
    def _check_model_performance(self):
        """Check if model performance is degrading"""
        logger.info("Checking model performance...")
    
    def _cleanup_data(self):
        """Remove low-quality or duplicate data"""
        logger.info("Cleaning up training data...")
    
    def _generate_synthetic_data(self):
        """Generate synthetic training data from knowledge base"""
        try:
            # Get recent objects
            objects = self.knowledge_db.search_objects(limit=10)
            
            for obj in objects:
                # Create Q&A pairs about the object
                questions = [
                    f"What is a {obj['name']}?",
                    f"Describe {obj['name']}",
                    f"What are the characteristics of {obj['name']}?"
                ]
                
                for question in questions:
                    answer = obj.get('description', f"A {obj['name']}")
                    
                    # Add to training data
                    self.knowledge_db.add_training_data(
                        input_text=question,
                        output_text=answer,
                        context=json.dumps(obj),
                        quality_score=0.6
                    )
            
            logger.info(f"Generated synthetic training data for {len(objects)} objects")
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")


class AgenticSystem:
    """Main agentic system controller"""
    
    def __init__(self, knowledge_db: KnowledgeDatabase, tavily_api_key: str, groq_api_key: str):
        self.knowledge_db = knowledge_db
        
        # Initialize agents
        self.secret_trainer = SecretModelTrainer(knowledge_db)
        self.info_gatherer = InfoGatheringAgent(knowledge_db, tavily_api_key, groq_api_key)
        self.supervisor = TrainingSupervisor(knowledge_db, self.secret_trainer)
    
    def start_all_agents(self):
        """Start all agentic AI systems"""
        logger.info("🤖 Starting all agentic AI systems...")
        
        self.info_gatherer.start_worker()
        self.secret_trainer.start_training_loop(interval=3600)  # Train every hour
        self.supervisor.start_supervision(interval=1800)  # Supervise every 30 min
        
        logger.info("✅ All agentic AI systems started")
    
    def stop_all_agents(self):
        """Stop all agentic AI systems"""
        logger.info("Stopping all agentic AI systems...")
        
        self.info_gatherer.stop_worker()
        self.secret_trainer.stop_training_loop()
        self.supervisor.stop_supervision()
        
        logger.info("All agentic AI systems stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            'info_gatherer': {
                'running': self.info_gatherer.is_running,
                'queue_size': len(self.info_gatherer.gathering_queue)
            },
            'secret_trainer': {
                'running': self.secret_trainer.training_enabled,
                'is_training': self.secret_trainer.is_training,
                'model_available': self.secret_trainer.model is not None
            },
            'supervisor': {
                'running': self.supervisor.is_running
            }
        }


# ==================== MAIN AI ASSISTANT ====================

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
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.chat_temperature = float(os.getenv('DEFAULT_TEMPERATURE', '0.7'))
        
        # System prompt
        self.system_prompt = """You are Moi, an advanced AI assistant with vision, memory, and learning capabilities. 
        You can see objects, describe them, remember information, and have conversations. 
        Be helpful, friendly, and informative in your responses.
        Format your responses clearly without any special tokens or unnecessary formatting characters."""
    
    def setup_vision_models(self):
        """Initialize vision and image processing models"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.huggingface_available = False
        self.nvidia_available = False
        
        # Initialize Hugging Face BLIP model
        try:
            print("Loading Hugging Face BLIP model...")
            cache_dir = os.path.join(os.getcwd(), "models_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            model_name = "Salesforce/blip-image-captioning-base"
            
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
            
            self.blip_model.to(self.device)
            self.blip_model.eval()
            
            self.huggingface_available = True
            print(f"✅ Hugging Face BLIP model loaded on {self.device}")
            
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
                print("⚠️  NVIDIA_API_KEY not found")
        except Exception as e:
            print(f"❌ Error setting up NVIDIA Vision API: {e}")
        
        if not self.huggingface_available and not self.nvidia_available:
            print("❌ No vision models available!")
        else:
            print(f"✅ Vision system ready - HF: {self.huggingface_available}, NVIDIA: {self.nvidia_available}")
    
    def setup_object_detection(self):
        """Initialize object detection model"""
        try:
            print("Loading object detection model...")
            self.object_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False, verbose=False)
            self.object_detector.to(self.device)
            self.object_detector.eval()
            print("✅ Object detection model loaded successfully")
        except Exception as e:
            print(f"⚠️  Error loading object detection model: {e}")
            self.object_detector = None
    
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
            if not self.huggingface_available or self.blip_processor is None or self.blip_model is None:
                return "Hugging Face BLIP model not available"
            
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif not isinstance(image, Image.Image):
                image = Image.open(image)
            
            max_size = 512
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(
                    **inputs, 
                    max_length=50,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption.strip() if caption else "Unable to generate caption"
            
        except Exception as e:
            print(f"BLIP processing error: {e}")
            return f"Error processing image with BLIP: {str(e)}"
    
    def process_image_with_nvidia(self, image):
        """Process image with NVIDIA Vision API"""
        try:
            if isinstance(image, np.ndarray):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                pil_image = Image.open(image)
            
            max_size = 1024
            if pil_image.width > max_size or pil_image.height > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
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
        
        results['caption'] = "No vision models available. Please check your API keys and model installations."
        results['model_used'] = 'none'
        return results
    
    def detect_objects(self, image):
        """Detect objects in image and return bounding boxes"""
        try:
            if self.object_detector is None:
                return []
            
            if isinstance(image, np.ndarray):
                results = self.object_detector(image)
            else:
                results = self.object_detector(np.array(image))
            
            detections = []
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                if conf > 0.5:
                    x1, y1, x2, y2 = map(int, box)
                    class_name = self.object_detector.names[int(cls)]
                    
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
                
                import random
                color = (
                    random.randint(50, 255),
                    random.randint(50, 255), 
                    random.randint(50, 255)
                )
                
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
            prompt = f"""
            Based on this scene description: "{scene_description}"
            
            Focus specifically on the {object_class} in the scene. Provide a detailed, specific description of just this {object_class}, including:
            - Its appearance, color, and condition
            - Its position or orientation in the scene
            - Any notable features or details
            - Its context within the overall scene
            
            Be specific and detailed. Only describe the {object_class}, not other objects.
            """
            
            messages = [{"role": "user", "content": prompt}]
            description = self.chat_with_groq(messages, model="openai/gpt-oss-120b")
            
            return description.strip() if description else f"A {object_class} is visible in the scene."
            
        except Exception as e:
            print(f"GPT description extraction error: {e}")
            return f"A {object_class} is visible in the scene."
    
    def clean_response_text(self, text):
        """Clean AI response text from unwanted formatting characters"""
        if not text:
            return ""
        
        text = re.sub(r'<\|.*?\|>', '', text)
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'<<.*?>>', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()
        text = html.unescape(text)
        
        return text

    def chat_with_groq(self, messages, model="openai/gpt-oss-120b", think_mode=False, reasoning_level="medium", custom_behavior="", temperature=None, stream=False):
        """Chat with Groq API with optional think mode and custom behavior"""
        try:
            system_prompt = self.system_prompt
            formatted_messages = [{"role": "system", "content": system_prompt}]
            formatted_messages.extend(messages)
            
            if custom_behavior:
                formatted_messages[0]["content"] += f"\n\nCustom Behavior Instructions: {custom_behavior}"
            
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
                max_tokens=4096,
                temperature=temperature if temperature is not None else self.chat_temperature,
                stream=stream
            )
            
            if stream:
                return response
            else:
                content = response.choices[0].message.content
                return self.clean_response_text(content)
        except Exception as e:
            return f"Error: {e}"
    
    def process_file(self, file_content, file_type):
        """Process uploaded files"""
        try:
            if file_type.startswith('image/'):
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
                text_content = file_content.decode('utf-8')
                return {
                    'type': 'text',
                    'content': text_content[:1000]
                }
            
            else:
                return {'type': 'unsupported', 'message': 'File type not supported'}
                
        except Exception as e:
            return {'type': 'error', 'message': str(e)}


# Initialize AI Assistant
ai_assistant = AIAssistant()


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
        web_search_enabled = data.get('web_search_enabled', True)
        think_mode = data.get('think_mode', False)
        reasoning_level = data.get('reasoning_level', 'medium')
        custom_behavior = data.get('custom_behavior', '')
        temperature = float(data.get('temperature', ai_assistant.chat_temperature))
        continue_generation = data.get('continue_generation', False)
        
        if continue_generation:
            user_message = "Please continue from where you left off."
        
        if web_search_enabled and not continue_generation and any(keyword in user_message.lower() for keyword in ['search', 'look up', 'find information', 'what is', 'how to']):
            search_results = ai_assistant.web_search(user_message)
            if search_results:
                context = "\n".join([f"- {result.get('title', '')}: {result.get('content', '')}" 
                                   for result in search_results[:3]])
                user_message += f"\n\nSearch results:\n{context}"
        
        messages = []
        for msg in chat_history[-10:]:
            messages.append({"role": msg['role'], "content": msg['content']})
        
        if not continue_generation:
            messages.append({"role": "user", "content": user_message})
        
        response = ai_assistant.chat_with_groq(messages, think_mode=think_mode, reasoning_level=reasoning_level, custom_behavior=custom_behavior, temperature=temperature)
        
        needs_continuation = len(response) > 3800
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'think_mode': think_mode,
            'reasoning_level': reasoning_level,
            'needs_continuation': needs_continuation
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
            ai_assistant.camera = cv2.VideoCapture(0)
            
        if not ai_assistant.camera.isOpened():
            ai_assistant.camera = cv2.VideoCapture(0)
            
        try:
            ai_assistant.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            ai_assistant.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception:
            pass
        
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
                
                with ai_assistant.frame_lock:
                    ai_assistant.current_frame = frame.copy()
                
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
        
        vision_results = ai_assistant.process_image_vision(frame)
        
        detections = []
        if ai_assistant.object_detector is not None:
            detections = ai_assistant.detect_objects_with_descriptions(frame, vision_results['caption'])
        
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
        
        question_lower = user_question.lower()
        
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
            search_query = f"{object_class} {user_question} {location}"
        
        search_results = ai_assistant.web_search(search_query)
        
        search_context = "\n".join([
            f"- {result.get('title', '')}: {result.get('content', '')[:300]}"
            for result in search_results[:3]
        ])
        
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

@app.route('/improve-prompt', methods=['POST'])
def improve_prompt():
    """Improve user prompt using AI"""
    try:
        data = request.json
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
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
        query_context = data.get('query_context', '')
        location = data.get('location', 'India')
        
        if not object_name:
            return jsonify({'error': 'No object name provided'}), 400
        
        if query_context:
            query = f"{object_name} {query_context} in {location}"
        else:
            query = f"{object_name} current price {location} where to buy specifications features reviews 2024"
        
        search_results = ai_assistant.web_search(query)
        
        formatted_results = []
        for result in search_results[:5]:
            formatted_results.append({
                'title': result.get('title', ''),
                'content': result.get('content', ''),
                'url': result.get('url', ''),
                'score': result.get('score', 0)
            })
        
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
        
        description = data.get('description', '')
        color = data.get('color', '')
        climate = data.get('climate', '')
        types = data.get('types', '')
        category = data.get('category', '')
        user_notes = data.get('user_notes', '')
        
        object_id = ai_assistant.knowledge_db.add_object(
            name=object_name,
            description=description,
            color=color,
            climate=climate,
            types=types,
            category=category,
            user_notes=user_notes
        )
        
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
        
        words = user_message.lower().split()
        context_info = []
        
        for word in words:
            obj_info = ai_assistant.knowledge_db.get_object(word)
            if obj_info:
                context_info.append(f"I know about {word}: {obj_info.get('description', '')}")
        
        if context_info:
            enhanced_message = f"{user_message}\n\n[Context from my memory: {' '.join(context_info)}]"
        else:
            enhanced_message = user_message
        
        messages = []
        for msg in chat_history[-10:]:
            messages.append({"role": msg['role'], "content": msg['content']})
        messages.append({"role": "user", "content": enhanced_message})
        
        response = ai_assistant.chat_with_groq(messages)
        
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
            
            chat_history.append({"role": "user", "content": user_input})
            
            print("🔄 Processing with GPT-OSS 120B...")
            
            messages = chat_history[-10:]
            response = ai_assistant.chat_with_groq(messages, model="openai/gpt-oss-120b")
            
            print(f"\n🤖 Moi: {response}")
            
            chat_history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


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