import time
import traceback
import numpy as np
import pandas as pd
from io import StringIO

from sqlalchemy.orm import Session
from src import (
    DEFAULT_CLASSIFIER,
    JOBS,
    LOGISTIC_REGRESSION_MODEL,
    SGD_CLASSIFIER_MODEL,
)


from src.db.crud.comments import list_all_comments
from src.db.crud.fileinfo import get_fileinfo
from src.db.crud.trainedmodel import get_trained_model_name
from src.db.schemas import TrainModelForm
from src.helper import (
    dict_to_lists,
    format_seconds,
    get_file_hash,
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
def process_data_and_train(
    job_id: str, filename: str, file_content: bytes, data: TrainModelForm, db: Session
):
    start_time = time.time()
    logger.info(f"[{job_id}] Background task started.")

    try:
        update_job(
            job_id,
            status="Processing",
            message="Decoding CSV and processing data...",
        )

        existing_model = get_trained_model_name(db, data.modelName)
        no_of_trained_data = 0

        if data.classifierModel == LOGISTIC_REGRESSION_MODEL:
            if existing_model:
                update_job(
                    job_id,
                    status="Error",
                    message=f"Model name {data.modelName} already exists.",
                )
                return

        logger.info(f"[{job_id}] Decoding CSV bytes")

        s = file_content.decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(s))
        logger.info(f"[{job_id}] CSV loaded | Rows: {len(df)}")

        file_id = get_file_hash(file_content)

        # Check if file already used for training
        if get_fileinfo(db, file_id):
            update_job(
                job_id,
                status="Error",
                message=f"File {filename} already used for training. You can delete the file for retraining.",
            )
            return

        # comments, labels, errors = process_data(df)
        result, errors = process_data(df)
        new_comments, new_labels = dict_to_lists(result)
        no_of_new_comments = len(new_comments)
        logger.info(f"no. of new comments: {no_of_new_comments}")

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
        combined_comments = result | {m.comment: m.label for m in retrieved_comments}
        all_comments, all_labels = dict_to_lists(combined_comments)

        # get the equivalent sentiments for the labels
        sentiments = convert_label_to_sentiment(all_labels)

        no_of_trained_data = len(all_comments)
        logger.info(f"no_of_trained_data: {no_of_trained_data}")

        # retrieve all comments from the database if logistic regression is used.
        # if data.classifierModel == LOGISTIC_REGRESSION_MODEL:
        #     retrieved_comments = list_all_comments(db)
        #     all_comments = comments + [m.to_dict() for m in retrieved_comments]
        #     no_of_trained_data = len(all_comments)
        # elif data.classifierModel == SGD_CLASSIFIER_MODEL:
        #     if existing_model:
        #         no_of_trained_data = existing_model.no_of_data + no_of_comments

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
        logger.info(f"[{job_id}] Training model using SGDClassifier")
        clf, report, accuracy = perform_training(
            job_id, data.modelName, data.classifierModel, X_train, y_train
        )
        logger.info(f"[{job_id}] Report: {report}")

        # ---------------------------------------
        # Save model and other information
        # ---------------------------------------
        save_file_info(db, file_id, filename, no_of_new_comments, errors)

        save_comments(db, file_id, new_comments, new_labels, sentiments)

        remarks = (
            f"Model is trained by {data.classifierModel} with {accuracy}% accuracy."
        )
        save_trained_model(
            db, clf, data, round(accuracy, 2), no_of_trained_data, remarks
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
