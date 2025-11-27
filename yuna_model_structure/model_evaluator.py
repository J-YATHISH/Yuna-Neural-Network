import logging
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate model performance."""

    @staticmethod
    def evaluate(model, X_test, y_test):
        """Evaluate model on test data."""
        logger.info("Evaluating model on test data...")

        # Get predictions
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.7).astype(int)  # Threshold at 0.7

        # Calculate metrics
        loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)

        logger.info("\nTest Results:")
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
            'loss': float(loss),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'confusion_matrix': cm.tolist()
        }
