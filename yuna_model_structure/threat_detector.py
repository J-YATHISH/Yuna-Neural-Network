import json
import logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


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
