"""
Agentic AI System for Moi AI Assistant
Contains three autonomous agents:
1. Secret Model Trainer - Continuously trains and refines a local model
2. Info Gathering Agent - Scrapes data when new objects are detected
3. Training Supervisor - Manages training loops and data quality
"""

import os
import time
import json
import threading
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from knowledge_database import KnowledgeDatabase
from groq import Groq
from tavily import TavilyClient
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            
            # Simple quality heuristics
            for item in training_data:
                quality_score = 0.5  # Default
                
                # Check length
                if len(item['input_text']) > 10 and len(item['output_text']) > 20:
                    quality_score += 0.2
                
                # Check if has context
                if item.get('context'):
                    quality_score += 0.2
                
                # Update quality score in database
                # (Would need to add update method to KnowledgeDatabase)
                
            logger.info(f"Evaluated {len(training_data)} training examples")
            
        except Exception as e:
            logger.error(f"Error evaluating training data: {e}")
    
    def _check_model_performance(self):
        """Check if model performance is degrading"""
        # Placeholder for model evaluation
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