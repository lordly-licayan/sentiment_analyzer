from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from src import DEFAULT_TEST_SIZE
from src.helper import logger, update_job


def create_logistic_regression(max_iter=1000, solver="lbfgs"):
    clf = LogisticRegression(max_iter=max_iter, solver=solver)
    return clf


def train_logistic_regression(
    job_id, clf, X, y, test_size=DEFAULT_TEST_SIZE, random_state=42
):
    """
    Train Logistic Regression model and update job status.
    Args:
        job_id (str): Job identifier for status updates
        X (np.array): Training features
        y (np.array or list): Training labels
        test_size (float): Fraction of data for validation
        random_state (int): Random seed for reproducibility
    Returns:
        report: Classification report dictionary
    """

    logger.info("Training using LogisticRegression model.")
    update_job(
        job_id,
        status="Training",
        message=f"Training {X.shape[0]} data using LogisticRegression...",
    )

    # Split train & validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf.fit(X_train, y_train)

    update_job(
        job_id,
        status="Evaluating",
        message="Evaluating accuracy of the model...",
    )

    # Evaluate on validation set
    y_val_pred = clf.predict(X_val)
    val_report = classification_report(y_val, y_val_pred, output_dict=True)
    val_acc = accuracy_score(y_val, y_val_pred)

    metrics = {
        "accuracy": round(val_acc * 100, 2),
        "precision": round(val_report["weighted avg"]["precision"] * 100, 2),
        "recall": round(val_report["weighted avg"]["recall"] * 100, 2),
        "f1_score": round(val_report["weighted avg"]["f1-score"] * 100, 2),
    }

    logger.info(f"Training complete. Report: {metrics}")

    update_job(
        job_id,
        message="Done training logistic regression model.",
        metrics=metrics,
    )
    return metrics
