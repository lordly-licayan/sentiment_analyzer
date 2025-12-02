import asyncio
import time
import traceback
import numpy as np
import pandas as pd

from src import (
    DEFAULT_CLASSIFIER,
    JOBS,
    LOGISTIC_REGRESSION_MODEL,
    SUPPORTED_CLASSIFIERS,
)


from src.db.crud.comments import list_all_comments
from src.db.crud.fileinfo import get_fileinfo
from src.db.crud.trainedmodel import get_trained_model_name
from src.db.database import get_db
from src.helper import (
    dict_to_lists,
    format_seconds,
    logger,
    process_data,
    get_embedder,
    convert_label_to_sentiment,
    save_comments,
    save_file_info,
    save_trained_model,
    update_job,
)
from src.model.logistic_regression import (
    create_logistic_regression,
    train_logistic_regression,
)
from src.model.sgd_classifier import get_model, train_sgd_classifier


def perform_embedding(job_id, comments):
    """
    Perform embedding on a list of comments with error handling.
    """
    try:
        embedder = get_embedder()
    except Exception as e:
        logger.error(f"Failed to load embedder: {e}")
        update_job(
            job_id, status="Error", progress="0%", message=f"Embedder load error: {e}"
        )
        raise

    batch_size = 64
    embeddings_list = []

    try:
        total_batches = len(comments) // batch_size + 1

        for batch_idx in range(total_batches):
            try:
                logger.info(f"Embedding batch {batch_idx + 1}/{total_batches}")

                batch = comments[batch_idx * batch_size : (batch_idx + 1) * batch_size]

                # Skip empty batch (happens if comments length is divisible by batch_size)
                if not batch:
                    continue

                batch_embeddings = embedder.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

                embeddings_list.append(batch_embeddings)

                # Report progress
                progress = int((batch_idx + 1) / total_batches * 100)
                logger.info(f"Embedding progress: {progress}%")

                update_job(
                    job_id,
                    progress=f"{progress}%",
                    message=f"Embedding batch {batch_idx + 1}/{total_batches}.",
                )

            except Exception as batch_error:
                logger.error(f"Error embedding batch {batch_idx + 1}: {batch_error}")
                update_job(
                    job_id,
                    status="Error",
                    message=f"Error embedding batch {batch_idx + 1}: {batch_error}",
                )
                # Optionally continue or raise → choose behavior:
                raise  # stop whole process

        # Final assembly
        return np.vstack(embeddings_list)

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        update_job(job_id, status="Error", message=f"Embedding failed: {e}")
        raise


def perform_training(job_id, model_name, classifier_model, X_train, y_train):
    """
    Perform training using the specified classifier.
    """
    if classifier_model == DEFAULT_CLASSIFIER:
        clf = create_logistic_regression()
        report, accuracy = train_logistic_regression(job_id, clf, X_train, y_train)
    else:
        clf = get_model(model_name)
        report, accuracy = train_sgd_classifier(job_id, clf, X_train, y_train)
    return clf, report, accuracy


# -----------------------------------------------------------
# BACKGROUND TASK (TRAINING)
# -----------------------------------------------------------
async def process_data_and_train(
    job_id: str, file_id: str, filename: str, df: pd.DataFrame, data: dict
):

    start_time = time.time()
    logger.info(f"[{job_id}] Background task started.")

    try:
        db = next(get_db())

        update_job(
            job_id,
            status="Processing",
            message="Decoding CSV and processing data...",
        )

        classifier_model = data.get("classifierModel")
        ext = SUPPORTED_CLASSIFIERS.get(classifier_model)

        model_name = f"{data.get('modelName')}.{ext}"

        existing_model = get_trained_model_name(db, model_name)

        if classifier_model == LOGISTIC_REGRESSION_MODEL:
            if existing_model:
                update_job(
                    job_id,
                    status="Error",
                    message=f"Model name {model_name} already exists.",
                )
                return

        logger.info(f"[{job_id}] Decoding CSV bytes")

        logger.info(f"[{job_id}] CSV loaded | Rows: {len(df)}")

        # Check if file already used for training
        if get_fileinfo(db, file_id):
            update_job(
                job_id,
                status="Error",
                message=f"File {filename} already used for training. You can delete the file for retraining.",
            )
            return

        result, errors = process_data(df)
        new_comments, new_labels = dict_to_lists(result)
        no_of_new_comments = len(new_comments)

        if no_of_new_comments == 0:
            logger.info(f"[{job_id}] No valid data to train on.")
            update_job(job_id, status="Complete", message="No valid data to train on.")
            return

        if len(errors) > 0:
            logger.info(f"[{job_id}] Data processing errors: {errors}")
            update_job(job_id, feedback=f"{errors}")

        # retrieve comments from the database
        retrieved_comments = list_all_comments(db)

        # combine retrieved comments and the new comments
        combined_comments = {m.comment: m.label for m in retrieved_comments} | result
        all_comments, all_labels = dict_to_lists(combined_comments)

        # get the equivalent sentiments for the labels
        sentiments = convert_label_to_sentiment(all_labels)
        no_of_trained_data = len(all_comments)

        # ---------------------------------------
        # Embedding
        # ---------------------------------------
        logger.info(f"[{job_id}] Embedding {no_of_trained_data} comments.")
        update_job(
            job_id,
            status="Embedding",
            message="Embedding the dataset started...",
        )
        X_train = perform_embedding(job_id, all_comments)
        y_train = np.array(sentiments)

        # ---------------------------------------
        # Train model
        # ---------------------------------------
        logger.info(f"[{job_id}] Training model.")
        clf, report, accuracy = perform_training(
            job_id, model_name, classifier_model, X_train, y_train
        )

        # ---------------------------------------
        # Save model and other information
        # ---------------------------------------
        save_file_info(db, file_id, filename, no_of_new_comments, errors)

        save_comments(db, file_id, new_comments, new_labels, sentiments)

        remarks = f"Model is trained by {classifier_model} with {accuracy}% accuracy."
        save_trained_model(
            db, clf, data, round(accuracy, 2), no_of_trained_data, remarks, model_name
        )

        elapsedTime = format_seconds(time.time() - start_time)
        logger.info(f"[{job_id}] Training completed in {elapsedTime}.")
        update_job(
            job_id,
            status="Complete",
            progress="100%",
            elapsedTime=elapsedTime,
            message="Training and saving the trained model done.",
        )

    except Exception as e:
        logger.error(f"[{job_id}] ERROR: {e}")
        logger.error(traceback.format_exc())

        JOBS[job_id]["status"] = "Error"
        JOBS[job_id]["message"] = str(e)


def run_trainer(job_id, file_id, filename, df, data):
    """Sync wrapper to run async function properly."""
    asyncio.run(process_data_and_train(job_id, file_id, filename, df, data))
