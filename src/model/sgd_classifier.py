import os
import joblib
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from src import JOBS
from src.helper import logger, update_job
from sklearn.metrics import classification_report, accuracy_score


def create_SGD_classifier():
    """
    Create a SGDClassifier with predefined parameters.
    """
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        eta0=0.001,
        max_iter=1,
        warm_start=True,
        tol=None,
        random_state=42,
    )

    return clf


def get_model(model_name: str):

    model_path = os.path.join(os.getenv("TRAINED_MODEL_DIR"), model_name)
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
    else:
        clf = create_SGD_classifier()
    return clf


def calculate_epochs(n_samples: int) -> int:
    """
    Automatically adjust number of epochs based on dataset size
    """
    if n_samples < 10000:
        return 15
    elif n_samples <= 50000:
        return 20
    elif n_samples <= 100000:
        return 25
    else:
        return 10  # For very large datasets, use fewer epochs with partial_fit


def train_sgd_classifier(job_id, clf, X, y, test_size=0.2):
    """
    Train SGDClassifier with training/validation split and early stopping.

    Args:
        job_id: ID for logging
        clf: SGDClassifier instance
        X: Features (embeddings)
        y: Labels
        test_size: Fraction for validation

    Returns:
        report: Classification report on validation set
    """

    # Split into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scale embeddings
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    n_samples = X_train_scaled.shape[0]
    epochs = calculate_epochs(n_samples)
    classes = np.unique(y_train)  # full set of labels

    logger.info(f"Job {job_id}: Starting training for {epochs} epochs")
    update_job(
        job_id,
        status="Training",
        message="Starting training for {epochs} epochs.",
    )

    # Early stopping variables
    best_acc = 0
    wait = 0
    patience = int(os.getenv("PATIENCE", 5))  # default 5 if not set

    for epoch in range(epochs):

        # Shuffle training data each epoch
        indices = np.random.permutation(n_samples)
        X_epoch = X_train_scaled[indices]
        y_epoch = y_train[indices]

        # Train 1 epoch
        clf.partial_fit(X_epoch, y_epoch, classes=classes)

        # Evaluate on validation set
        y_val_pred = clf.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_val_pred)
        progress = int((epoch + 1) / epochs * 100)
        logger.info(
            f"Epoch {epoch+1}/{epochs} - Val Accuracy: {acc:.4f} - Progress: {progress}%"
        )
        update_job(
            job_id,
            message=f"Epoch {epoch+1}/{epochs} - Accuracy: {acc * 100:.2f}% - Progress: {progress}%",
        )
        # Early stopping
        if acc > best_acc:
            best_acc = acc
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            logger.warning("⏹ Early stopping triggered")
            break

        # update_job(
        #     job_id,
        #     progress=f"{progress}%",
        #     accuracy=f"{acc * 100:.2f}%",
        #     message=f"Training epoch {epoch+1}/{epochs}.",
        # )
    # Final evaluation on validation set
    report = classification_report(y_val, y_val_pred, output_dict=True)
    logger.info(f"Training completed. Best Val Accuracy: {best_acc:.4f}")

    update_job(
        job_id,
        status="Complete",
        progress=f"{progress}%",
        accuracy=f"{acc * 100:.2f}%",
        message="Training complete.",
        report=report,
    )

    return report
