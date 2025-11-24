import os
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
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


def train_sgd_classifier(job_id, clf, X_train, y_train, batch_size=64):
    """
    Train an SGDClassifier incrementally using partial_fit.

    Args:
        clf: An instance of SGDClassifier.
        comments (list[str]): List of text comments.
        labels (list): Corresponding labels for the comments.
        batch_size (int): Size of each training batch.
    """

    n_samples = X_train.shape[0]
    epochs = calculate_epochs(n_samples)
    classes = np.unique(y_train)  # full set of labels
    logger.info(
        f"Classes: {classes}  | Epochs: {epochs}  | Samples: {n_samples}  | Batch size: {batch_size}  "
    )
    update_job(
        job_id,
        status="Training",
        message="Data count: {n_samples}, Epochs: {epochs}.",
    )

    # Early stopping variables
    best_acc = 0
    wait = 0
    patience = int(os.getenv("PATIENCE"))

    for epoch in range(epochs):

        # Shuffle each epoch
        indices = np.random.permutation(n_samples)
        X_epoch = X_train[indices]
        y_epoch = y_train[indices]

        # partial_fit = 1 epoch of SGD
        clf.partial_fit(X_epoch, y_epoch, classes=classes)

        # Evaluate on this batch
        y_pred = clf.predict(X_train)
        acc = accuracy_score(y_train, y_pred)
        progress = int((epoch + 1) / epochs * 100)

        logger.info(
            f"[Epoch {epoch+1}/{epochs}] Progress: {progress}% | Accuracy: {acc:.4f}"
        )

        # Early stopping
        if acc > best_acc:
            best_acc = acc
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            print("⏹ Early stopping triggered")
            break

        update_job(
            job_id,
            progress=f"{progress}%",
            accuracy=f"{acc * 100:.2f}%",
            message=f"Training epoch {epoch+1}/{epochs}.",
        )
        print(JOBS[job_id])

    # Final report
    report = classification_report(y_train, y_pred, output_dict=True)
    logger.info("Training complete!")
    update_job(
        job_id,
        status="Complete",
        message="Training complete.",
        report=report,
    )

    return report
