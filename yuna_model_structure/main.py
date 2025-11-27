import json
import logging

import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_preparator import DataPreparator
from threat_detection_model import ThreatDetectionModel
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from threat_detector import ThreatDetector

# Configure logging once here
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Complete training pipeline."""

    logger.info("=" * 70)
    logger.info("YUNA FIREWALL - NEURAL NETWORK THREAT DETECTION TRAINING")
    logger.info("=" * 70)

    # Phase 1: Data Preparation
    logger.info("\n[PHASE 1] DATA PREPARATION")
    preparator = DataPreparator()

    # Generate or load dataset
    df = preparator.generate_synthetic_dataset(n_samples=10000)

    # Normalize features
    df_normalized = preparator.normalize_features(df)

    # Prepare data
    X, y = preparator.prepare_data(df_normalized)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # Save normalization parameters
    preparator.save_normalization_params()

    # Phase 2: Build Model
    logger.info("\n[PHASE 2] NETWORK ARCHITECTURE DESIGN")
    model_builder = ThreatDetectionModel()
    model = model_builder.build_model(input_dim=4)
    logger.info(model_builder.get_model_info())

    # Phase 3: Train Model
    logger.info("\n[PHASE 3] TRAINING PROCESS")
    trainer = ModelTrainer(model)
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=20, batch_size=64, patience=3
    )

    # Plot training history
    trainer.plot_training_history()

    # Evaluate Model
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
    result1 = detector.detect(
        packet_rate=50,
        avg_packet_size=1.5,
        connection_duration=2.0,
        port_number=443
    )
    logger.info(f"Normal traffic: {result1}")

    # Test case 2: DDoS attack
    result2 = detector.detect(
        packet_rate=900,
        avg_packet_size=0.05,
        connection_duration=0.5,
        port_number=8080
    )
    logger.info(f"DDoS attack: {result2}")

    # Test case 3: Ransomware-like
    result3 = detector.detect(
        packet_rate=400,
        avg_packet_size=3.0,
        connection_duration=5.0,
        port_number=445
    )
    logger.info(f"Ransomware-like: {result3}")


if __name__ == "__main__":
    # Optional: configure GPU growth if you are using GPU
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    main()
