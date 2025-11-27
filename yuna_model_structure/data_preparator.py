import numpy as np
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


class DataPreparator:
    """Handles data loading, normalization, and preprocessing."""

    def __init__(self):
        self.normalization_params = {
            'packet_rate_max': 1000.0,      # packets/sec
            'packet_size_max': 10.0,        # MB
            'duration_max': 24.0,           # hours
            'port_max': 65535.0             # port number
        }

    def generate_synthetic_dataset(self, n_samples=10000):
        """
        Generate synthetic training data for demonstration.
        In production, replace this with real packet capture data.
        """
        logger.info(f"Generating {n_samples} synthetic samples...")

        np.random.seed(42)
        data = []

        # Generate normal traffic (60% of data)
        n_normal = int(n_samples * 0.6)
        for _ in range(n_normal):
            data.append({
                'packet_rate': np.random.uniform(1, 100),        # Low rate
                'avg_packet_size': np.random.uniform(0.1, 2.0),  # Normal size
                'connection_duration': np.random.uniform(0.01, 5.0),  # Short to medium
                'port_number': np.random.choice([80, 443, 22, 21, 25, 53]),  # Common ports
                'label': 0  # Normal
            })

        # Generate DDoS attacks (15% of data)
        n_ddos = int(n_samples * 0.15)
        for _ in range(n_ddos):
            data.append({
                'packet_rate': np.random.uniform(500, 1000),     # Very high rate
                'avg_packet_size': np.random.uniform(0.01, 0.1), # Very small packets
                'connection_duration': np.random.uniform(0.1, 2.0),
                'port_number': np.random.randint(1, 65535),
                'label': 1  # Threat
            })

        # Generate Port Scans (10% of data)
        n_portscan = int(n_samples * 0.10)
        for _ in range(n_portscan):
            data.append({
                'packet_rate': np.random.uniform(100, 300),
                'avg_packet_size': np.random.uniform(0.05, 0.2),
                'connection_duration': np.random.uniform(0.001, 0.1),  # Very short
                'port_number': np.random.randint(1, 65535),
                'label': 1  # Threat
            })

        # Generate Ransomware-like activity (10% of data)
        n_ransomware = int(n_samples * 0.10)
        for _ in range(n_ransomware):
            data.append({
                'packet_rate': np.random.uniform(200, 600),
                'avg_packet_size': np.random.uniform(0.5, 5.0),   # Larger packets
                'connection_duration': np.random.uniform(1.0, 10.0),  # Longer duration
                'port_number': np.random.choice([445, 139, 3389, 135]),  # SMB, RDP ports
                'label': 1  # Threat
            })

        # Generate remaining threats (5% of data)
        n_other = n_samples - n_normal - n_ddos - n_portscan - n_ransomware
        for _ in range(n_other):
            data.append({
                'packet_rate': np.random.uniform(150, 800),
                'avg_packet_size': np.random.uniform(0.1, 8.0),
                'connection_duration': np.random.uniform(0.1, 20.0),
                'port_number': np.random.randint(1, 65535),
                'label': 1  # Threat
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
