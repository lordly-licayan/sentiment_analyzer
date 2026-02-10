import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


from src import DEFAULT_TEST_SIZE, RANDOM_STATE
from src.classifiers.classifier_trainer import ClassifierTrainer
from src.helper import logger, update_job


class SVMClassifierModel(ClassifierTrainer):
    def __init__(self, model_name=None):
        """Initialize the classifier."""

        self.clf = self._create_SVM_classifier()
        self.model_name = model_name

    def _create_SVM_classifier(self) -> LinearSVC:
        """
        Create a LinearSVC with predefined parameters.
        """
        self.clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000)
        return self.clf

    def get_classifier(self):
        return self.clf

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(
        self,
        job_id,
        X,
        y,
        comments,
        test_size=DEFAULT_TEST_SIZE,
        random_state=RANDOM_STATE,
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

        logger.info("Training using SVM Classifier model.")
        update_job(
            job_id,
            status="Training",
            message=f"Training {X.shape[0]} data using SVM Classifier...",
        )

        # Split train & validation
        X_train, X_val, y_train, y_val, texts_train, texts_val = train_test_split(
            X, y, comments, test_size=test_size, random_state=random_state, stratify=y
        )

        self.clf.fit(X_train, y_train)

        update_job(
            job_id,
            status="Evaluating",
            message="Evaluating accuracy of the model...",
        )

        # Evaluate on validation set
        y_val_pred = self.clf.predict(X_val)
        decision_scores = self.clf.decision_function(X_val)
        val_report = classification_report(y_val, y_val_pred, output_dict=True)
        val_acc = accuracy_score(y_val, y_val_pred)

        evaluation_results = []
        for i, (comment, true_label, pred_label) in enumerate(
            zip(texts_val, y_val, y_val_pred)
        ):
            raw_confidence = float(np.max(decision_scores[i]))
            confidence_pct = float(round(self._sigmoid(raw_confidence) * 100, 2))

            evaluation_results.append(
                {
                    "comment": comment,
                    "actual_label": str(true_label),
                    "predicted_label": str(pred_label),
                    "is_matched": str(true_label) == str(pred_label),
                    "confidence": confidence_pct,
                }
            )

        metrics = {
            "accuracy": round(val_acc * 100, 2),
            "weighted_average": {
                "precision": round(val_report["weighted avg"]["precision"] * 100, 2),
                "recall": round(val_report["weighted avg"]["recall"] * 100, 2),
                "f1_score": round(val_report["weighted avg"]["f1-score"] * 100, 2),
            },
            "macro_average": {
                "precision": round(val_report["macro avg"]["precision"] * 100, 2),
                "recall": round(val_report["macro avg"]["recall"] * 100, 2),
                "f1_score": round(val_report["macro avg"]["f1-score"] * 100, 2),
            },
        }

        logger.info(f"Training complete. Report: {metrics}")

        update_job(
            job_id,
            message="Done training logistic regression model.",
            metrics=metrics,
        )
        return metrics, evaluation_results
