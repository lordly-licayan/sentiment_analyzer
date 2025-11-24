from fastapi import HTTPException
import os
import time
import joblib
import traceback
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, accuracy_score

from io import StringIO

import tqdm
from model.classifier import create_SGD_classifier
from src import JOBS, MODEL_PATH
from src.helper import (
    get_model,
    logger,
    process_data,
    get_embedder,
    convert_label_to_sentiment,
    get_test_data,
    remove_job,
    update_job,
)


def evaluate_model(model_path, csv_path):
    import joblib
    from sklearn.metrics import accuracy_score, classification_report
    import pandas as pd

    # Load model
    clf = joblib.load(model_path)

    # Load test dataset
    df = pd.read_csv(csv_path)
    comments = df["comment"].tolist()
    labels = df["label"].tolist()

    # Embeddings
    embedder = get_embedder()
    X = embedder.encode(comments, batch_size=64, show_progress_bar=True)

    # Predict
    y_pred = clf.predict(X)

    # Metrics
    acc = accuracy_score(labels, y_pred)
    report = classification_report(labels, y_pred, output_dict=True)

    return acc, report


def get_embeddings(comments):
    """
    Get embeddings for a list of comments using the embedder model.
    """
    embedder = get_embedder()
    embeddings = embedder.encode(
        comments, batch_size=64, show_progress_bar=False, convert_to_numpy=True
    )
    return embeddings


def perform_embedding(job_id, comments):
    """
    Perform embedding on a list of comments.
    """
    embedder = get_embedder()

    batch_size = 64
    embeddings_list = []
    total_batches = len(comments) // batch_size + 1

    for batch_idx in range(total_batches):
        logger.info(f"Embedding batch {batch_idx + 1}/{total_batches}")

        batch = comments[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_embeddings = embedder.encode(
            batch, convert_to_numpy=True, show_progress_bar=False
        )
        embeddings_list.append(batch_embeddings)

        # Report progress (0-100)
        progress = int((batch_idx + 1) / total_batches * 100)
        logger.info(f"Embedding progress: {progress}%")
        update_job(
            job_id,
            progress=f"{progress}%",
            message=f"Embedding batch {batch_idx + 1}/{total_batches}.",
        )

    # Combine into single NumPy array
    embeddings = np.vstack(embeddings_list)

    return embeddings


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


def train_SGDClassifier(job_id, clf, X_train, y_train, batch_size=64):
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

    # ------------------------------------------------
    # 4. Training Loop (REAL-TIME PROGRESS)
    # ------------------------------------------------

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
        progress="100%",
        message="Training complete",
        report=report,
    )

    return report


# -----------------------------------------------------------
# BACKGROUND TASK (TRAINING)
# -----------------------------------------------------------
def process_data_and_train(job_id: str, modelName: str, content: bytes):
    start_time = time.time()
    logger.info(f"[{job_id}] Background task started.")

    try:
        update_job(
            job_id,
            status="Processing",
            message="Decoding CSV and processing data...",
        )

        logger.info(f"[{job_id}] Decoding CSV bytes")

        s = content.decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(s))
        logger.info(f"[{job_id}] CSV loaded | Rows: {len(df)}")

        comments, labels, errors = process_data(df)

        if len(comments) == 0:
            logger.info(f"[{job_id}] No valid data to train on.")
            update_job(job_id, status="Complete", message="No valid data to train on.")
            return

        if len(errors) > 0:
            logger.info(f"[{job_id}] Data processing errors: {errors}")
            update_job(job_id, feedback=f"{errors}")

        # ---------------------------------------
        # Embedding
        # ---------------------------------------
        logger.info(f"[{job_id}] Embedding {len(comments)} comments")
        update_job(
            job_id,
            status="Embedding",
            message="Start embedding the dataset...",
        )
        X_train = perform_embedding(job_id, comments)
        y_train = np.array(convert_label_to_sentiment(labels))

        # ---------------------------------------
        # Train model
        # ---------------------------------------
        logger.info(f"[{job_id}] Training model using SGDClassifier")
        clf = get_model(modelName)
        report = train_SGDClassifier(job_id, clf, X_train, y_train)
        logger.info(f"[{job_id}] Report: {report}")

        # ---------------------------------------
        # Save model
        # ---------------------------------------
        logger.info(f"[{job_id}] Saving model → {MODEL_PATH}")
        joblib.dump(clf, MODEL_PATH)

        elapsed = time.time() - start_time
        logger.info(f"[{job_id}] Training completed in {elapsed:.2f} seconds")

    except Exception as e:
        logger.error(f"[{job_id}] ERROR: {e}")
        logger.error(traceback.format_exc())

        JOBS[job_id]["status"] = "Error"
        JOBS[job_id]["message"] = str(e)
