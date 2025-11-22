import time
import joblib
import traceback
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from io import StringIO
from src import JOBS, MODEL_PATH
from src.helper import (
    logger,
    process_data,
    get_embedder,
    convert_label_to_sentiment,
    get_test_data,
)


# -----------------------------------------------------------
# BACKGROUND TASK (TRAINING)
# -----------------------------------------------------------
def process_data_and_train(job_id: str, content: bytes):
    start_time = time.time()
    logger.info(f"[{job_id}] Background task started.")

    try:
        JOBS[job_id]["status"] = "processing"
        logger.info(f"[{job_id}] Decoding CSV bytes")

        s = content.decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(s))
        logger.info(f"[{job_id}] CSV loaded | Rows: {len(df)}")

        comments, labels, errors = process_data(df)

        if errors:
            logger.warning(f"[{job_id}] Found {len(errors)} rows with issues")

        # ---------------------------------------
        # Embedding
        # ---------------------------------------
        logger.info(f"[{job_id}] Embedding {len(comments)} comments")

        embedder = get_embedder()
        embeddings = embedder.encode(comments, show_progress_bar=True)
        X_train = np.array(embeddings)
        y_train = convert_label_to_sentiment(labels)

        # ---------------------------------------
        # Train model
        # ---------------------------------------
        logger.info(f"[{job_id}] Training LogisticRegression")
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_train, y_train)

        # ---------------------------------------
        # Evaluate model
        logger.info(f"[{job_id}] Evaluating model")
        # ---------------------------------------
        test_data_comments, test_data_labels, _ = get_test_data()
        X_test = embedder.encode(test_data_comments)
        y_test = convert_label_to_sentiment(test_data_labels)

        preds = clf.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)

        # ---------------------------------------
        # Save model
        # ---------------------------------------
        logger.info(f"[{job_id}] Saving model → {MODEL_PATH}")
        joblib.dump({"clf": clf, "label_order": list(clf.classes_)}, MODEL_PATH)

        # ---------------------------------------
        # Mark job completed
        # ---------------------------------------
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = {
            "n_samples": len(comments),
            "report": report,
            "classes": list(clf.classes_),
        }

        elapsed = time.time() - start_time
        logger.info(f"[{job_id}] Training completed in {elapsed:.2f} seconds")

    except Exception as e:
        logger.error(f"[{job_id}] ERROR: {e}")
        logger.error(traceback.format_exc())

        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["detail"] = str(e) + "\n" + traceback.format_exc()
