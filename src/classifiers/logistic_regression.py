from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from src.helper import logger, update_job


def create_logistic_regression(max_iter=1000, solver="lbfgs"):
    clf = LogisticRegression(max_iter=max_iter, solver=solver)
    return clf


def train_logistic_regression(
    job_id, clf, X, y, test_size=0.2, random_state=42, max_iter=1000
):
    """
    Train Logistic Regression model and update job status.
    Args:
        job_id (str): Job identifier for status updates
        X (np.array): Training features
        y (np.array or list): Training labels
        test_size (float): Fraction of data for validation
        random_state (int): Random seed for reproducibility
        max_iter (int): Maximum iterations for Logistic Regression
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

    # Scale embeddings
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train
    clf.fit(X_train_scaled, y_train)

    update_job(
        job_id,
        status="Evaluating",
        message="Evaluating accuracy of the model...",
    )

    # Evaluate on training set
    y_train_pred = clf.predict(X_train_scaled)
    train_report = classification_report(y_train, y_train_pred, output_dict=True)
    train_acc = accuracy_score(y_train, y_train_pred)

    # Evaluate on validation set
    y_val_pred = clf.predict(X_val_scaled)
    val_report = classification_report(y_val, y_val_pred, output_dict=True)
    val_acc = accuracy_score(y_val, y_val_pred)
    accuracy = round(val_acc * 100, 2)

    report = {
        "trained_report": train_report,
        "trained_accuracy": train_acc,
        "validation_report": val_report,
        "validation_accuracy": val_acc,
    }

    logger.info(f"Training complete. Report: {report}")

    update_job(
        job_id,
        accuracy=f"{accuracy:.2f}%",
        message="Done training logistic regression model.",
        report=report,
    )
    return report, accuracy
