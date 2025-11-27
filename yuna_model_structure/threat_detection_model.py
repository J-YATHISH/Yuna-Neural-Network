import logging
import tensorflow as tf
from tensorflow.keras import layers, models

logger = logging.getLogger(__name__)


class ThreatDetectionModel:
    """Neural network model for threat detection."""

    def __init__(self):
        self.model = None
        self.history = None

    def build_model(self, input_dim=4):
        """
        Build the neural network architecture.

        Architecture:
        - Input: 4 features
        - Hidden Layer 1: 32 neurons (ReLU)
        - Dropout: 20%
        - Hidden Layer 2: 16 neurons (ReLU)
        - Output: 1 neuron (Sigmoid) - threat probability
        """
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
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
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
