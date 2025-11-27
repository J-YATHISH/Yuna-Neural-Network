import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json
import logging

# Configure Logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreparator:
    
    def __init__(self):
        self.normalization_params = {
            'packet_rate_max': 1000.0,   
            'packet_size_max': 10.0,         
            'duration_max': 24.0,           
            'port_max': 65535.0              
        }
    
    def generate_synthetic_dataset(self, n_samples=10000):
        logger.info(f"Generating {n_samples} synthetic samples...")
        
        np.random.seed(42)
        data = []
        
        n_normal = int(n_samples * 0.6)
        for _ in range(n_normal):
            data.append({
                'packet_rate': np.random.uniform(1, 100),        
                'avg_packet_size': np.random.uniform(0.1, 2.0),  
                'connection_duration': np.random.uniform(0.01, 5.0),  
                'port_number': np.random.choice([80, 443, 22, 21, 25, 53]),  
                'label': 0  
            })
        

        n_ddos = int(n_samples * 0.15)
        for _ in range(n_ddos):
            data.append({
                'packet_rate': np.random.uniform(500, 1000),     
                'avg_packet_size': np.random.uniform(0.01, 0.1), 
                'connection_duration': np.random.uniform(0.1, 2.0),
                'port_number': np.random.randint(1, 65535),
                'label': 1 
            })
        

        n_portscan = int(n_samples * 0.10)
        for _ in range(n_portscan):
            data.append({
                'packet_rate': np.random.uniform(100, 300),
                'avg_packet_size': np.random.uniform(0.05, 0.2),
                'connection_duration': np.random.uniform(0.001, 0.1),  
                'port_number': np.random.randint(1, 65535),
                'label': 1  
            })
        
        n_ransomware = int(n_samples * 0.10)
        for _ in range(n_ransomware):
            data.append({
                'packet_rate': np.random.uniform(200, 600),
                'avg_packet_size': np.random.uniform(0.5, 5.0),  
                'connection_duration': np.random.uniform(1.0, 10.0),  
                'port_number': np.random.choice([445, 139, 3389, 135]), 
                'label': 1  
            })
        

        n_other = n_samples - n_normal - n_ddos - n_portscan - n_ransomware
        for _ in range(n_other):
            data.append({
                'packet_rate': np.random.uniform(150, 800),
                'avg_packet_size': np.random.uniform(0.1, 8.0),
                'connection_duration': np.random.uniform(0.1, 20.0),
                'port_number': np.random.randint(1, 65535),
                'label': 1
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Dataset generated: {len(df)} samples")
        logger.info(f"Normal: {(df['label']==0).sum()}, Threats: {(df['label']==1).sum()}")
        
        return df
    
    def normalize_features(self, df):
        """Normalize features to [0, 1] range."""
        logger.info("Normalizing features...")
        
        df_normalized = df.copy()
        df_normalized['packet_rate'] = df['packet_rate'] / self.normalization_params['packet_rate_max']
        df_normalized['avg_packet_size'] = df['avg_packet_size'] / self.normalization_params['packet_size_max']
        df_normalized['connection_duration'] = df['connection_duration'] / self.normalization_params['duration_max']
        df_normalized['port_number'] = df['port_number'] / self.normalization_params['port_max']
        
        return df_normalized
    
    def prepare_data(self, df):
        """Split into features (X) and labels (y)."""
        feature_columns = ['packet_rate', 'avg_packet_size', 'connection_duration', 'port_number']
        X = df[feature_columns].values
        y = df['label'].values
        
        logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        return X, y
    
    def save_normalization_params(self, filename='normalization_params.json'):
        """Save normalization parameters for production use."""
        with open(filename, 'w') as f:
            json.dump(self.normalization_params, f, indent=4)
        logger.info(f"Normalization parameters saved to {filename}")

class ThreatDetectionModel:
    
    def __init__(self):
        self.model = None
        self.history = None
    
    def build_model(self, input_dim=4):
        logger.info("Building neural network model...")
        
        self.model = models.Sequential([
            # Input layer + First hidden layer
            layers.Dense(32, activation='relu', input_shape=(input_dim,), name='hidden_layer_1'),
            
            # Dropout for regularization
            layers.Dropout(0.2, name='dropout'),
            
            # Second hidden layer
            layers.Dense(16, activation='relu', name='hidden_layer_2'),
            
            # Output layer
            layers.Dense(1, activation='sigmoid', name='output_layer')
        ])
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall']
        )
        
        logger.info("Model architecture:")
        self.model.summary()
        
        return self.model
    
    def get_model_info(self):
        """Return model architecture information."""
        if self.model is None:
            return "Model not built yet"
        
        total_params = self.model.count_params()
        return f"Total parameters: {total_params:,}"

class ModelTrainer:
    
    def __init__(self, model):
        self.model = model
        self.history = None
    
    def train(self, X_train, y_train, X_val, y_val,epochs=20, batch_size=64, patience=3):
        
        
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Model checkpoint callback
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        logger.info("Training completed!")
        return self.history
    
    
    '''def plot_training_history(self, save_path='training_history.png'):
        """Plot and save training history."""
        if self.history is None:
            logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
        plt.close() '''

class ModelEvaluator:
    
    @staticmethod
    def evaluate(model, X_test, y_test):
        logger.info("Evaluating model on test data...")
        
        # Get predictions
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.7).astype(int)  # Threshold at 0.7
        
        # Calculate metrics
        loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"\nTest Results:")
        logger.info(f"Loss: {loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        
        # Classification report
        logger.info("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Threat']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        logger.info(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm.tolist()
        }

#Real time detection

class ThreatDetector:
    """Production-ready threat detector for YUNA Firewall."""
    
    def __init__(self, model_path='yuna_threat_model.h5', 
                 norm_params_path='normalization_params.json'):
        """Load trained model and normalization parameters."""
        logger.info("Loading threat detection model...")
        self.model = tf.keras.models.load_model(model_path)
        
        with open(norm_params_path, 'r') as f:
            self.norm_params = json.load(f)
        
        self.threat_threshold = 0.7
        logger.info("Threat detector ready!")
    
    def normalize_features(self, packet_rate, avg_packet_size, 
                          connection_duration, port_number):
        """Normalize raw features."""
        normalized = [
            packet_rate / self.norm_params['packet_rate_max'],
            avg_packet_size / self.norm_params['packet_size_max'],
            connection_duration / self.norm_params['duration_max'],
            port_number / self.norm_params['port_max']
        ]
        return np.array(normalized).reshape(1, -1)
    
    def detect(self, packet_rate, avg_packet_size, connection_duration, port_number):
        """
        Detect if given connection features indicate a threat.
        
        Returns:
            dict: {
                'is_threat': bool,
                'probability': float,
                'decision': str
            }
        """
        # Normalize features
        features = self.normalize_features(
            packet_rate, avg_packet_size, 
            connection_duration, port_number
        )
        
        # Predict
        probability = self.model.predict(features, verbose=0)[0][0]
        is_threat = probability > self.threat_threshold
        
        return {
            'is_threat': bool(is_threat),
            'probability': float(probability),
            'decision': 'BLOCK' if is_threat else 'ALLOW'
        }

def main():
    """Complete training pipeline."""
    
    logger.info("=" * 70)
    logger.info("YUNA FIREWALL - NEURAL NETWORK THREAT DETECTION TRAINING")
    logger.info("=" * 70)
    

    logger.info("\n[PHASE 1] DATA PREPARATION")
    preparator = DataPreparator()
    

    df = preparator.generate_synthetic_dataset(n_samples=10000)
    

    df_normalized = preparator.normalize_features(df)

    X, y = preparator.prepare_data(df_normalized)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    preparator.save_normalization_params()
    

    logger.info("\n[PHASE 2] NETWORK ARCHITECTURE DESIGN")
    model_builder = ThreatDetectionModel()
    model = model_builder.build_model(input_dim=4)
    logger.info(model_builder.get_model_info())
    

    logger.info("\n[PHASE 3] TRAINING PROCESS")
    trainer = ModelTrainer(model)
    history = trainer.train(X_train, y_train, X_val, y_val, 
                           epochs=20, batch_size=64, patience=3)
    logger.info("\n[PHASE 4] MODEL EVALUATION")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(model, X_test, y_test)
    
    # Save final model
    model.save('yuna_threat_model.h5')
    logger.info("\nModel saved as 'yuna_threat_model.h5'")
    
    # Save results
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    logger.info("Results saved to 'training_results.json'")
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 70)
    
    # Demo: Real-time inference
    logger.info("\n[DEMO] REAL-TIME THREAT DETECTION")
    detector = ThreatDetector()
    
    # Test case 1: Normal traffic
    result1 = detector.detect(packet_rate=50, avg_packet_size=1.5,connection_duration=2.0, port_number=443)
    
    logger.info(f"Normal traffic: {result1}")
    
    # Test case 2: DDoS attack
    result2 = detector.detect(packet_rate=900, avg_packet_size=0.05,connection_duration=0.5, port_number=8080)
    
    logger.info(f"DDoS attack: {result2}")
    
    # Test case 3: Ransomware-like
    result3 = detector.detect(packet_rate=400, avg_packet_size=3.0, connection_duration=5.0, port_number=445)
    
    logger.info(f"Ransomware-like: {result3}")

if __name__ == "__main__":
    main()
