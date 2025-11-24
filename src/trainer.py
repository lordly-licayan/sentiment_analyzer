import os
import time
import joblib
import traceback
import numpy as np
import pandas as pd
from io import StringIO
from src import (
    DEFAULT_CLASSIFIER,
    JOBS,
    SUPPORTED_CLASSIFIERS,
    TRAINED_MODEL_DIR,
    DEFAULT_TRAINED_MODEL_NAME,
)
from src.helper import (
    logger,
    process_data,
    get_embedder,
    convert_label_to_sentiment,
    update_job,
)
from src.model.logistic_regression import (
    create_logistic_regression,
    train_logistic_regression,
)
from src.model.sgd_classifier import get_model, train_sgd_classifier


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


def perform_training(job_id, model_name, classifier_model, X_train, y_train):
    """
    Perform training using the specified classifier.
    """
    if classifier_model == DEFAULT_CLASSIFIER:
        clf = create_logistic_regression()
        report = train_logistic_regression(job_id, clf, X_train, y_train)
    else:
        clf = get_model(model_name)
        report = train_sgd_classifier(job_id, clf, X_train, y_train)
    return clf, report


def save_model(
    job_id,
    clf,
    classifier_model,
    model_name=DEFAULT_TRAINED_MODEL_NAME,
    model_dir=TRAINED_MODEL_DIR,
):
    """
    Save the trained model to disk.
    """
    ext = SUPPORTED_CLASSIFIERS.get(classifier_model)
    # SUPPORTED_CLASSIFIERS[classifier_model]
    model_path = os.path.join(model_dir, f"{model_name}.{ext}")
    joblib.dump(clf, model_path)
    logger.info(f"Model saved to {model_path}")

    update_job(
        job_id,
        status="Saving",
        message=f"Trained model {model_name} saved.",
    )


# -----------------------------------------------------------
# BACKGROUND TASK (TRAINING)
# -----------------------------------------------------------
def process_data_and_train(
    job_id: str, model_name: str, classifier_model: str, content: bytes
):
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
            message="Embedding the dataset started...",
        )
        X_train = perform_embedding(job_id, comments)
        y_train = np.array(convert_label_to_sentiment(labels))

        # ---------------------------------------
        # Train model
        # ---------------------------------------
        logger.info(f"[{job_id}] Training model using SGDClassifier")
        clf, report = perform_training(
            job_id, model_name, classifier_model, X_train, y_train
        )
        logger.info(f"[{job_id}] Report: {report}")

        # ---------------------------------------
        # Save model
        # ---------------------------------------
        save_model(job_id, clf, classifier_model, model_name)

        elapsed = time.time() - start_time
        logger.info(f"[{job_id}] Training completed in {elapsed:.2f} seconds")
        update_job(
            job_id,
            status="Complete",
            progress="100%",
            message="Training and saving the trained model done.",
        )

    except Exception as e:
        logger.error(f"[{job_id}] ERROR: {e}")
        logger.error(traceback.format_exc())

        JOBS[job_id]["status"] = "Error"
        JOBS[job_id]["message"] = str(e)
