"""Database module for storing training data and references in Neon PostgreSQL."""

import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

import psycopg2
from psycopg2.extras import Json, execute_values
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class NeonDB:
    """Interface to Neon PostgreSQL database for training data tracking."""

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize database connection.
        
        Args:
            connection_string: PostgreSQL connection string. If None, reads from env var NEON_DB_URL.
        """
        self.connection_string = connection_string or os.getenv("NEON_DB_URL")
        if not self.connection_string:
            raise ValueError("Database connection string not provided. Set NEON_DB_URL environment variable.")
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = psycopg2.connect(self.connection_string)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def initialize_schema(self):
        """Create all necessary tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Training runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id SERIAL PRIMARY KEY,
                    mlflow_run_id VARCHAR(255) UNIQUE,
                    model_type VARCHAR(50) NOT NULL,
                    model_name VARCHAR(255),
                    training_started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    training_completed_at TIMESTAMP,
                    num_training_examples INTEGER,
                    num_validation_examples INTEGER,
                    hyperparameters JSONB,
                    metrics JSONB,
                    status VARCHAR(50) DEFAULT 'running',
                    notes TEXT
                )
            """)
            
            # Training data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    data_id SERIAL PRIMARY KEY,
                    run_id INTEGER REFERENCES training_runs(run_id) ON DELETE CASCADE,
                    claim TEXT NOT NULL,
                    evidence TEXT,
                    label VARCHAR(50) NOT NULL,
                    data_type VARCHAR(20) DEFAULT 'train',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_data_run_id 
                ON training_data(run_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_data_label 
                ON training_data(label)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_runs_mlflow 
                ON training_runs(mlflow_run_id)
            """)
            
            logger.info("Database schema initialized successfully")

    def create_training_run(
        self,
        model_type: str,
        model_name: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Create a new training run record.
        
        Args:
            model_type: Type of model (e.g., 'bert', 't5', 'gan')
            model_name: Name of the model
            hyperparameters: Training hyperparameters as dict
            notes: Optional notes about the training run
            
        Returns:
            run_id: ID of the created training run
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_runs 
                (model_type, model_name, hyperparameters, notes)
                VALUES (%s, %s, %s, %s)
                RETURNING run_id
            """, (model_type, model_name, Json(hyperparameters or {}), notes))
            
            run_id = cursor.fetchone()[0]
            logger.info(f"Created training run {run_id} for {model_type}")
            return run_id

    def update_training_run(
        self,
        run_id: int,
        mlflow_run_id: Optional[str] = None,
        num_training_examples: Optional[int] = None,
        num_validation_examples: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ):
        """Update training run with additional information.
        
        Args:
            run_id: ID of the training run
            mlflow_run_id: MLflow run ID to reference
            num_training_examples: Number of training examples used
            num_validation_examples: Number of validation examples used
            metrics: Training metrics as dict
            status: Status of training ('running', 'completed', 'failed')
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            updates = []
            values = []
            
            if mlflow_run_id is not None:
                updates.append("mlflow_run_id = %s")
                values.append(mlflow_run_id)
            
            if num_training_examples is not None:
                updates.append("num_training_examples = %s")
                values.append(num_training_examples)
            
            if num_validation_examples is not None:
                updates.append("num_validation_examples = %s")
                values.append(num_validation_examples)
            
            if metrics is not None:
                updates.append("metrics = %s")
                values.append(Json(metrics))
            
            if status is not None:
                updates.append("status = %s")
                values.append(status)
                
                if status == 'completed':
                    updates.append("training_completed_at = CURRENT_TIMESTAMP")
            
            if updates:
                values.append(run_id)
                query = f"UPDATE training_runs SET {', '.join(updates)} WHERE run_id = %s"
                cursor.execute(query, values)
                logger.info(f"Updated training run {run_id}")

    def save_training_data(
        self,
        run_id: int,
        data: List[Dict[str, Any]],
        data_type: str = 'train',
        batch_size: int = 1000,
    ):
        """Save training data in batches.
        
        Args:
            run_id: ID of the training run
            data: List of training examples, each with 'claim', 'evidence', 'label'
            data_type: Type of data ('train' or 'validation')
            batch_size: Number of rows to insert per batch
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare data for batch insert
            values = [
                (run_id, item['claim'], item.get('evidence', ''), item['label'], data_type)
                for item in data
            ]
            
            # Insert in batches
            for i in range(0, len(values), batch_size):
                batch = values[i:i + batch_size]
                execute_values(
                    cursor,
                    """
                    INSERT INTO training_data 
                    (run_id, claim, evidence, label, data_type)
                    VALUES %s
                    """,
                    batch
                )
                logger.info(f"Saved batch {i//batch_size + 1} ({len(batch)} examples)")
            
            logger.info(f"Saved {len(data)} {data_type} examples for run {run_id}")

    def get_training_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get training run details by ID.
        
        Args:
            run_id: ID of the training run
            
        Returns:
            Dict with training run details or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    run_id, mlflow_run_id, model_type, model_name,
                    training_started_at, training_completed_at,
                    num_training_examples, num_validation_examples,
                    hyperparameters, metrics, status, notes
                FROM training_runs
                WHERE run_id = %s
            """, (run_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                'run_id': row[0],
                'mlflow_run_id': row[1],
                'model_type': row[2],
                'model_name': row[3],
                'training_started_at': row[4],
                'training_completed_at': row[5],
                'num_training_examples': row[6],
                'num_validation_examples': row[7],
                'hyperparameters': row[8],
                'metrics': row[9],
                'status': row[10],
                'notes': row[11],
            }

    def get_training_run_by_mlflow(self, mlflow_run_id: str) -> Optional[Dict[str, Any]]:
        """Get training run details by MLflow run ID.
        
        Args:
            mlflow_run_id: MLflow run ID
            
        Returns:
            Dict with training run details or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    run_id, mlflow_run_id, model_type, model_name,
                    training_started_at, training_completed_at,
                    num_training_examples, num_validation_examples,
                    hyperparameters, metrics, status, notes
                FROM training_runs
                WHERE mlflow_run_id = %s
            """, (mlflow_run_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                'run_id': row[0],
                'mlflow_run_id': row[1],
                'model_type': row[2],
                'model_name': row[3],
                'training_started_at': row[4],
                'training_completed_at': row[5],
                'num_training_examples': row[6],
                'num_validation_examples': row[7],
                'hyperparameters': row[8],
                'metrics': row[9],
                'status': row[10],
                'notes': row[11],
            }

    def get_training_data_stats(self, run_id: int) -> Dict[str, Any]:
        """Get statistics about training data for a run.
        
        Args:
            run_id: ID of the training run
            
        Returns:
            Dict with data statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Overall counts
            cursor.execute("""
                SELECT data_type, label, COUNT(*) as count
                FROM training_data
                WHERE run_id = %s
                GROUP BY data_type, label
            """, (run_id,))
            
            stats = {
                'train': {},
                'validation': {},
            }
            
            for row in cursor.fetchall():
                data_type, label, count = row
                stats[data_type][label] = count
            
            return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    db = NeonDB()
    db.initialize_schema()
    print("Database schema initialized")
