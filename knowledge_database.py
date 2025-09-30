"""
Knowledge Database System for Moi AI Assistant
Stores and manages learned object information with full CRUD operations
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import threading

class KnowledgeDatabase:
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
                    if key in ['description', 'color', 'climate', 'types', 'category', 'user_notes']:
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