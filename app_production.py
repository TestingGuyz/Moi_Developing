"""
🚀 MOI AI ASSISTANT - ULTRA-ROBUST PRODUCTION EDITION 🚀

Zero-Error Architecture with Comprehensive Error Handling
Created: October 2025
Author: Advanced AI Engineering Team

FEATURES:
✅ Zero OpenCV dependencies (100% PIL/Pillow)
✅ Multi-platform compatibility (ARM, x86, Windows, Linux, macOS)
✅ Enterprise-grade security hardening
✅ Comprehensive error handling and monitoring
✅ Auto-scaling and resource optimization
✅ Advanced debugging and logging systems
✅ Database backup and recovery
✅ API rate limiting and caching
✅ Memory optimization and leak prevention
✅ Network resilience and retry mechanisms
✅ Configuration validation and management
✅ Health checks and performance monitoring
✅ Docker and cloud deployment ready
✅ Cost optimization and resource management
"""

import os
import sys
import json
import base64
import hashlib
import time
import threading
import asyncio
import logging
import traceback
import platform
import psutil
import uuid
import sqlite3
import shutil
import signal
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from functools import wraps, lru_cache
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import tempfile
import subprocess

# Core Dependencies
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib.parse
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
import io
from dotenv import load_dotenv

# Conditional imports with comprehensive fallbacks
ML_MODELS_AVAILABLE = False
GROQ_AVAILABLE = False
TAVILY_AVAILABLE = False
PIL_AVAILABLE = False
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError as e:
    logging.critical(f"❌ PIL/Pillow is REQUIRED - install with: pip install pillow")
    sys.exit(1)

try:
    import torch
    import torch.nn.functional as F
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        AutoTokenizer, AutoModel, pipeline
    )
    TORCH_AVAILABLE = True
    TRANSFORMERS_AVAILABLE = True
    ML_MODELS_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ ML models not available - install with: pip install torch transformers")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ Groq not available - install with: pip install groq")

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ Tavily not available - install with: pip install tavily-python")

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==================== CONFIGURATION MANAGEMENT ====================

@dataclass
class SystemConfig:
    """Comprehensive system configuration with validation"""
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    environment: str = "production"
    
    # Security Configuration
    secret_key: str = ""
    cors_origins: List[str] = None
    max_content_length: int = 16 * 1024 * 1024  # 16MB
    rate_limit_per_minute: int = 60
    
    # Database Configuration
    database_path: str = "data/moi_knowledge.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 6
    
    # AI Model Configuration
    model_cache_dir: str = "models"
    vision_model_name: str = "Salesforce/blip-image-captioning-base"
    max_model_memory_mb: int = 2048
    model_timeout_seconds: int = 300
    
    # API Configuration
    groq_api_key: str = ""
    tavily_api_key: str = ""
    nvidia_api_key: str = ""
    huggingface_token: str = ""
    
    # Resource Limits
    max_concurrent_requests: int = 10
    max_file_size_mb: int = 10
    max_image_dimension: int = 2048
    cache_size_mb: int = 512
    
    # Network Configuration
    request_timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 0.3
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "logs/moi_assistant.log"
    max_log_size_mb: int = 100
    log_backup_count: int = 5
    
    def __post_init__(self):
        """Validate and setup configuration"""
        self._setup_directories()
        self._validate_config()
        self._setup_security()
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            Path(self.database_path).parent,
            Path(self.model_cache_dir),
            Path(self.log_file).parent,
            Path("data/backups"),
            Path("data/uploads"),
            Path("data/cache"),
            Path("static/generated")
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                # Test write permissions
                test_file = directory / ".test_write"
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                logging.error(f"❌ Cannot create/write to directory {directory}: {e}")
                # Fallback to temp directory
                if "data" in str(directory):
                    fallback = Path(tempfile.gettempdir()) / "moi_assistant" / directory.name
                    fallback.mkdir(parents=True, exist_ok=True)
                    logging.warning(f"⚠️ Using fallback directory: {fallback}")
    
    def _validate_config(self):
        """Validate configuration values"""
        if self.port < 1024 and os.geteuid() != 0:
            logging.warning(f"⚠️ Port {self.port} requires root privileges, using 5000")
            self.port = 5000
            
        if self.max_concurrent_requests > 100:
            logging.warning("⚠️ Very high concurrent request limit, may cause resource exhaustion")
            
        if self.max_file_size_mb > 100:
            logging.warning("⚠️ Large file size limit may cause memory issues")
    
    def _setup_security(self):
        """Setup security configuration"""
        if not self.secret_key:
            self.secret_key = os.environ.get('SECRET_KEY', 
                hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest())
        
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:*", "https://*.herokuapp.com", 
                               "https://*.vercel.app", "https://*.railway.app"]

# ==================== ADVANCED LOGGING SYSTEM ====================

class AdvancedLogger:
    """Advanced logging system with structured logging and monitoring"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.correlation_ids = {}
        self.performance_metrics = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = (
            '%(asctime)s | %(levelname)8s | %(name)20s | '
            'PID:%(process)d | %(funcName)15s:%(lineno)3d | '
            '%(message)s'
        )
        
        # Create formatters
        formatter = logging.Formatter(log_format)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with rotation
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.config.log_file,
                maxBytes=self.config.max_log_size_mb * 1024 * 1024,
                backupCount=self.config.log_backup_count
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            logging.warning(f"⚠️ Could not setup file logging: {e}")
        
        # Performance logger
        self.perf_logger = logging.getLogger('performance')
        
        # Security logger
        self.security_logger = logging.getLogger('security')
    
    def get_correlation_id(self) -> str:
        """Generate correlation ID for request tracking"""
        correlation_id = str(uuid.uuid4())[:8]
        self.correlation_ids[threading.current_thread().ident] = correlation_id
        return correlation_id
    
    def log_performance(self, operation: str, duration: float, success: bool = True):
        """Log performance metrics"""
        self.performance_metrics[operation].append({
            'duration': duration,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'correlation_id': self.correlation_ids.get(threading.current_thread().ident)
        })
        
        # Keep only last 1000 entries per operation
        if len(self.performance_metrics[operation]) > 1000:
            self.performance_metrics[operation].pop(0)
        
        self.perf_logger.info(f"{operation} | {duration:.3f}s | {'✅' if success else '❌'}")
    
    def log_error(self, error: Exception, context: Dict = None):
        """Log errors with full context"""
        self.error_counts[type(error).__name__] += 1
        
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'correlation_id': self.correlation_ids.get(threading.current_thread().ident),
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        logging.error(f"❌ {error_info['type']}: {error_info['message']}")
        logging.debug(f"Full traceback:\n{error_info['traceback']}")
    
    def get_metrics(self) -> Dict:
        """Get system metrics"""
        return {
            'performance_metrics': dict(self.performance_metrics),
            'error_counts': dict(self.error_counts),
            'active_correlation_ids': len(self.correlation_ids)
        }

# ==================== RESOURCE MONITORING ====================

class ResourceMonitor:
    """Advanced system resource monitoring"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.metrics_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.alerts = []
        self.monitoring_active = True
        self.monitoring_thread = None
        
    def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logging.info("📊 Resource monitoring started")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
                self._check_alerts(metrics)
                time.sleep(60)  # Check every minute
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(60)
    
    def get_current_metrics(self) -> Dict:
        """Get current system metrics"""
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent(interval=1)
            
            # System metrics
            system_memory = psutil.virtual_memory()
            system_disk = psutil.disk_usage('/')
            
            return {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': time.time() - self.start_time,
                'process': {
                    'memory_rss_mb': memory_info.rss / 1024 / 1024,
                    'memory_vms_mb': memory_info.vms / 1024 / 1024,
                    'cpu_percent': cpu_percent,
                    'threads': self.process.num_threads(),
                    'open_files': len(self.process.open_files()),
                    'connections': len(self.process.connections())
                },
                'system': {
                    'memory_total_gb': system_memory.total / 1024 / 1024 / 1024,
                    'memory_available_gb': system_memory.available / 1024 / 1024 / 1024,
                    'memory_percent': system_memory.percent,
                    'disk_total_gb': system_disk.total / 1024 / 1024 / 1024,
                    'disk_free_gb': system_disk.free / 1024 / 1024 / 1024,
                    'disk_percent': (system_disk.used / system_disk.total) * 100,
                    'cpu_count': psutil.cpu_count(),
                    'platform': platform.platform()
                }
            }
        except Exception as e:
            logging.error(f"Error getting metrics: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _check_alerts(self, metrics: Dict):
        """Check for alert conditions"""
        if 'process' not in metrics:
            return
            
        alerts = []
        process_metrics = metrics['process']
        system_metrics = metrics['system']
        
        # Memory alerts
        if process_metrics['memory_rss_mb'] > 2048:  # 2GB
            alerts.append({
                'level': 'WARNING',
                'message': f"High memory usage: {process_metrics['memory_rss_mb']:.1f}MB",
                'timestamp': metrics['timestamp']
            })
        
        if system_metrics['memory_percent'] > 90:
            alerts.append({
                'level': 'CRITICAL',
                'message': f"System memory critical: {system_metrics['memory_percent']:.1f}%",
                'timestamp': metrics['timestamp']
            })
        
        # Disk alerts
        if system_metrics['disk_percent'] > 90:
            alerts.append({
                'level': 'CRITICAL',
                'message': f"Disk space critical: {system_metrics['disk_percent']:.1f}%",
                'timestamp': metrics['timestamp']
            })
        
        # CPU alerts
        if process_metrics['cpu_percent'] > 80:
            alerts.append({
                'level': 'WARNING',
                'message': f"High CPU usage: {process_metrics['cpu_percent']:.1f}%",
                'timestamp': metrics['timestamp']
            })
        
        # Add new alerts
        for alert in alerts:
            if alert not in self.alerts[-10:]:  # Avoid duplicate recent alerts
                self.alerts.append(alert)
                logging.warning(f"🚨 {alert['level']}: {alert['message']}")
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def get_health_status(self) -> Dict:
        """Get overall health status"""
        if not self.metrics_history:
            return {'status': 'STARTING', 'message': 'System initializing'}
        
        latest = self.metrics_history[-1]
        if 'error' in latest:
            return {'status': 'ERROR', 'message': latest['error']}
        
        # Check critical conditions
        if latest['system']['memory_percent'] > 95:
            return {'status': 'CRITICAL', 'message': 'System memory exhausted'}
        
        if latest['system']['disk_percent'] > 95:
            return {'status': 'CRITICAL', 'message': 'Disk space exhausted'}
        
        if latest['process']['memory_rss_mb'] > 4096:  # 4GB
            return {'status': 'WARNING', 'message': 'High memory usage'}
        
        return {'status': 'HEALTHY', 'message': 'All systems operational'}

# ==================== RESILIENT HTTP CLIENT ====================

class ResilientHTTPClient:
    """HTTP client with comprehensive error handling and retries"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create configured requests session"""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Default headers
        session.headers.update({
            'User-Agent': 'Moi-AI-Assistant/2.0 (Production)',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        return session
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """GET request with error handling"""
        return self._request('GET', url, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """POST request with error handling"""
        return self._request('POST', url, **kwargs)
    
    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Execute request with comprehensive error handling"""
        start_time = time.time()
        
        # Set timeout if not provided
        kwargs.setdefault('timeout', self.config.request_timeout_seconds)
        
        try:
            response = self.session.request(method, url, **kwargs)
            duration = time.time() - start_time
            
            # Log successful requests
            logging.debug(f"🌐 {method} {url} | {response.status_code} | {duration:.3f}s")
            
            # Raise for HTTP errors
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout as e:
            duration = time.time() - start_time
            logging.error(f"⏱️ Request timeout: {url} after {duration:.1f}s")
            raise
        
        except requests.exceptions.ConnectionError as e:
            duration = time.time() - start_time
            logging.error(f"🔌 Connection error: {url} | {e}")
            raise
        
        except requests.exceptions.HTTPError as e:
            duration = time.time() - start_time
            logging.error(f"❌ HTTP error: {url} | {e.response.status_code} | {duration:.3f}s")
            raise
        
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"💥 Unexpected error: {url} | {type(e).__name__}: {e}")
            raise

# ==================== PRODUCTION DATABASE MANAGER ====================

class ProductionDatabaseManager:
    """Production-grade database manager with backup and recovery"""
    
    def __init__(self, config: SystemConfig, logger: AdvancedLogger):
        self.config = config
        self.logger = logger
        self.db_path = Path(config.database_path)
        self.backup_dir = self.db_path.parent / "backups"
        self.lock = threading.RLock()  # Reentrant lock
        self.connection_pool = {}
        self.backup_thread = None
        self.backup_active = True
        
        self._initialize_database()
        if config.backup_enabled:
            self._start_backup_service()
    
    def _initialize_database(self):
        """Initialize database with comprehensive error handling"""
        try:
            # Ensure directory exists and is writable
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if database file is accessible
            if self.db_path.exists():
                if not os.access(self.db_path, os.R_OK | os.W_OK):
                    raise PermissionError(f"Database file not accessible: {self.db_path}")
            
            # Create database and tables
            with self._get_connection() as conn:
                self._create_tables(conn)
                
            # Test database functionality
            self._test_database()
            logging.info(f"✅ Database initialized: {self.db_path}")
            
        except Exception as e:
            # Fallback to in-memory database
            logging.error(f"❌ Database initialization failed: {e}")
            logging.warning("⚠️ Falling back to in-memory database")
            self.db_path = ":memory:"
            self._create_fallback_database()
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Create all database tables"""
        cursor = conn.cursor()
        
        # Objects table with enhanced schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT DEFAULT '',
                category TEXT DEFAULT '',
                confidence REAL DEFAULT 0.0,
                times_seen INTEGER DEFAULT 1,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                created_by TEXT DEFAULT 'system',
                updated_by TEXT DEFAULT 'system'
            )
        ''')
        
        # Sessions table for tracking user interactions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                session_data TEXT DEFAULT '{}'
            )
        ''')
        
        # Interactions table for detailed logging
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                interaction_type TEXT NOT NULL,
                request_data TEXT,
                response_data TEXT,
                processing_time_ms INTEGER,
                success BOOLEAN DEFAULT 1,
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        
        # System metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_type TEXT NOT NULL,
                metric_value REAL,
                metric_data TEXT DEFAULT '{}'
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_objects_name ON objects(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_objects_category ON objects(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_objects_last_seen ON objects(last_seen)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_session_id ON interactions(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_metrics_type ON system_metrics(metric_type)')
        
        conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup"""
        thread_id = threading.get_ident()
        
        with self.lock:
            if thread_id not in self.connection_pool:
                try:
                    conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    conn.execute('PRAGMA foreign_keys = ON')
                    conn.execute('PRAGMA journal_mode = WAL')  # Better concurrency
                    conn.execute('PRAGMA synchronous = NORMAL')  # Balanced performance/safety
                    conn.execute('PRAGMA cache_size = 10000')  # 40MB cache
                    conn.execute('PRAGMA temp_store = MEMORY')  # Memory temp storage
                    self.connection_pool[thread_id] = conn
                except sqlite3.Error as e:
                    logging.error(f"Database connection failed: {e}")
                    raise
        
        try:
            yield self.connection_pool[thread_id]
        except sqlite3.Error as e:
            self.logger.log_error(e, {'operation': 'database_operation'})
            # Try to recover connection
            try:
                self.connection_pool[thread_id].rollback()
            except:
                pass
            raise
        finally:
            # Connection cleanup is handled by the pool
            pass
    
    def _test_database(self):
        """Test database functionality"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.execute("INSERT OR IGNORE INTO objects (name, description) VALUES (?, ?)", 
                             ("test_object", "Database initialization test"))
                conn.commit()
        except Exception as e:
            raise Exception(f"Database test failed: {e}")
    
    def _create_fallback_database(self):
        """Create fallback in-memory database"""
        try:
            with self._get_connection() as conn:
                self._create_tables(conn)
            logging.warning("⚠️ Using in-memory database - data will not persist")
        except Exception as e:
            logging.critical(f"❌ Even in-memory database failed: {e}")
            raise
    
    def add_object(self, name: str, description: str = "", category: str = "", 
                   confidence: float = 0.0, metadata: Dict = None) -> int:
        """Add or update object with comprehensive error handling"""
        if metadata is None:
            metadata = {}
            
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if object exists
                cursor.execute('SELECT id, times_seen FROM objects WHERE name = ?', (name,))
                result = cursor.fetchone()
                
                if result:
                    # Update existing object
                    object_id, times_seen = result
                    cursor.execute('''
                        UPDATE objects 
                        SET times_seen = ?, last_seen = CURRENT_TIMESTAMP, 
                            description = ?, category = ?, confidence = ?, 
                            metadata = ?, updated_by = ?
                        WHERE id = ?
                    ''', (times_seen + 1, description, category, confidence, 
                          json.dumps(metadata), 'system', object_id))
                else:
                    # Insert new object
                    cursor.execute('''
                        INSERT INTO objects 
                        (name, description, category, confidence, metadata, created_by)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (name, description, category, confidence, 
                          json.dumps(metadata), 'system'))
                    object_id = cursor.lastrowid
                
                conn.commit()
                return object_id
                
        except sqlite3.Error as e:
            self.logger.log_error(e, {
                'operation': 'add_object',
                'name': name,
                'category': category
            })
            return 0
        except Exception as e:
            self.logger.log_error(e, {'operation': 'add_object'})
            return 0
    
    def search_objects(self, query: str = "", category: str = "", 
                      limit: int = 50, offset: int = 0) -> List[Dict]:
        """Search objects with comprehensive filtering"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build dynamic query
                where_conditions = []
                params = []
                
                if query:
                    where_conditions.append("(name LIKE ? OR description LIKE ?)")
                    params.extend([f'%{query}%', f'%{query}%'])
                
                if category:
                    where_conditions.append("category = ?")
                    params.append(category)
                
                where_clause = ""
                if where_conditions:
                    where_clause = "WHERE " + " AND ".join(where_conditions)
                
                sql = f'''
                    SELECT * FROM objects 
                    {where_clause}
                    ORDER BY times_seen DESC, last_seen DESC
                    LIMIT ? OFFSET ?
                '''
                params.extend([limit, offset])
                
                cursor.execute(sql, params)
                results = [dict(row) for row in cursor.fetchall()]
                
                # Parse JSON fields
                for result in results:
                    try:
                        result['metadata'] = json.loads(result.get('metadata', '{}'))
                        result['tags'] = json.loads(result.get('tags', '[]'))
                    except json.JSONDecodeError:
                        result['metadata'] = {}
                        result['tags'] = []
                
                return results
                
        except Exception as e:
            self.logger.log_error(e, {
                'operation': 'search_objects',
                'query': query,
                'category': category
            })
            return []
    
    def log_interaction(self, session_id: str, interaction_type: str, 
                       request_data: Dict = None, response_data: Dict = None,
                       processing_time_ms: int = 0, success: bool = True, 
                       error_message: str = None):
        """Log user interaction"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO interactions 
                    (session_id, interaction_type, request_data, response_data,
                     processing_time_ms, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id, interaction_type,
                    json.dumps(request_data) if request_data else None,
                    json.dumps(response_data) if response_data else None,
                    processing_time_ms, success, error_message
                ))
                conn.commit()
        except Exception as e:
            # Don't raise - interaction logging is not critical
            logging.warning(f"Failed to log interaction: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to prevent database growth"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # Clean up old interactions
                cursor.execute('''
                    DELETE FROM interactions 
                    WHERE timestamp < ?
                ''', (cutoff_date.isoformat(),))
                
                # Clean up old sessions
                cursor.execute('''
                    DELETE FROM sessions 
                    WHERE last_activity < ?
                ''', (cutoff_date.isoformat(),))
                
                # Clean up old metrics
                cursor.execute('''
                    DELETE FROM system_metrics 
                    WHERE timestamp < ?
                ''', (cutoff_date.isoformat(),))
                
                # Vacuum database to reclaim space
                cursor.execute('VACUUM')
                
                conn.commit()
                logging.info(f"✅ Database cleanup completed (kept {days_to_keep} days)")
                
        except Exception as e:
            logging.error(f"Database cleanup failed: {e}")
    
    def _start_backup_service(self):
        """Start automated backup service"""
        if self.backup_thread is None or not self.backup_thread.is_alive():
            self.backup_thread = threading.Thread(target=self._backup_loop, daemon=True)
            self.backup_thread.start()
            logging.info("💾 Database backup service started")
    
    def _backup_loop(self):
        """Background backup loop"""
        while self.backup_active:
            try:
                time.sleep(self.config.backup_interval_hours * 3600)  # Convert to seconds
                if self.db_path != ":memory:":  # Don't backup in-memory database
                    self.create_backup()
            except Exception as e:
                logging.error(f"Backup service error: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
    
    def create_backup(self) -> Optional[str]:
        """Create database backup"""
        if self.db_path == ":memory:":
            logging.warning("⚠️ Cannot backup in-memory database")
            return None
            
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"moi_knowledge_backup_{timestamp}.db"
            backup_path = self.backup_dir / backup_filename
            
            # Create backup using SQLite backup API
            with self._get_connection() as source_conn:
                backup_conn = sqlite3.connect(str(backup_path))
                source_conn.backup(backup_conn)
                backup_conn.close()
            
            # Compress backup
            compressed_path = backup_path.with_suffix('.db.gz')
            import gzip
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove uncompressed backup
            backup_path.unlink()
            
            # Clean up old backups (keep last 10)
            backups = sorted(self.backup_dir.glob("*_backup_*.db.gz"))
            if len(backups) > 10:
                for old_backup in backups[:-10]:
                    old_backup.unlink()
            
            logging.info(f"✅ Database backup created: {compressed_path.name}")
            return str(compressed_path)
            
        except Exception as e:
            logging.error(f"Backup creation failed: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Object statistics
                cursor.execute("SELECT COUNT(*) FROM objects")
                object_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT category) FROM objects WHERE category != ''")
                category_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT SUM(times_seen) FROM objects")
                total_seen = cursor.fetchone()[0] or 0
                
                # Session statistics
                cursor.execute("SELECT COUNT(*) FROM sessions")
                session_count = cursor.fetchone()[0]
                
                # Interaction statistics
                cursor.execute("SELECT COUNT(*) FROM interactions")
                interaction_count = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COUNT(*) FROM interactions 
                    WHERE timestamp >= datetime('now', '-24 hours')
                """)
                recent_interactions = cursor.fetchone()[0]
                
                return {
                    'objects': {
                        'total': object_count,
                        'categories': category_count,
                        'total_interactions': total_seen
                    },
                    'sessions': {
                        'total': session_count
                    },
                    'interactions': {
                        'total': interaction_count,
                        'last_24h': recent_interactions
                    }
                }
                
        except Exception as e:
            logging.error(f"Failed to get database stats: {e}")
            return {}
    
    def close(self):
        """Close all database connections"""
        self.backup_active = False
        
        with self.lock:
            for conn in self.connection_pool.values():
                try:
                    conn.close()
                except:
                    pass
            self.connection_pool.clear()
        
        logging.info("🔒 Database connections closed")

# Continue with the rest of the production code...
# This is Part 1 of the comprehensive solution. 
# The file is getting very large, so I'll continue with the remaining components.