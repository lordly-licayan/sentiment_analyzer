from datetime import datetime
import re
import logging
import os
import joblib

import pandas as pd
from sentence_transformers import SentenceTransformer
from model.classifier import create_SGD_classifier
from src import EMBEDDER_MODEL, JOBS, TEST_DATA_PATH
from uuid import uuid4

# -----------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def read_data(path):
    """
    Load CSV with columns: 'comment' and 'label'.

    Behavior:
        - Removes commas from comments
        - Raises an error (skips row) when a comment contains a floating number
        - Skips rows with errors (empty comment, empty label, invalid label)
        - Collects error messages with row numbers
        - Returns (comments, labels, errors)
    """
    df = pd.read_csv(path)
    return df


def process_data(df: pd.DataFrame):
    """
    Process DataFrame with columns: 'comment' and 'label'.
    Behavior:
        - Removes commas from comments
        - Raises an error (skips row) when a comment contains a floating number
        - Skips rows with errors (empty comment, empty label, invalid label)
        - Collects error messages with row numbers
        - Returns (comments, labels, errors)
        Args:
        df (pd.DataFrame): DataFrame with 'comment' and 'label' columns
    """

    # --- Validate column presence ---
    if "comment" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'comment' and 'label' columns.")

    comments = []
    labels = []
    errors = []

    # --- Process rows one-by-one ---
    for idx, row in df.iterrows():
        row_num = idx + 1

        raw_comment = row["comment"]
        raw_label = row["label"]

        # --- Validate comment ---
        if pd.isna(raw_comment) or str(raw_comment).strip() == "":
            errors.append(f"Row {row_num}: Empty or missing comment.")
            continue

        comment = str(raw_comment).strip()

        if comment.lower() in ["nan", "none"]:
            errors.append(f"Row {row_num}: Comment is 'nan' or 'None'.")
            continue

        # --- Remove commas ---
        comment = comment.replace(",", "")

        # --- Check for floating number (e.g., 1.0, 3.14) ---
        if re.search(r"\b\d+\.\d+\b", comment):
            errors.append(f"Row {row_num}: Comment contains a floating number.")
            continue

        # --- Validate label ---
        if pd.isna(raw_label) or str(raw_label).strip() == "":
            errors.append(f"Row {row_num}: Empty or missing label.")
            continue

        try:
            label = int(raw_label)
        except ValueError:
            errors.append(f"Row {row_num}: Label is not an integer.")
            continue

        if label not in (-1, 0, 1):
            errors.append(f"Row {row_num}: Label must be -1, 0, or 1 (got {label}).")
            continue

        # --- Normalize comment ---
        def normalize(text):
            text = text.lower()
            text = re.sub(r"\s+", " ", text)
            return text

        comment = normalize(comment)

        # --- Store valid row ---
        comments.append(comment)
        labels.append(label)

    # If absolutely no valid rows exist, raise an error
    if len(comments) == 0:
        raise ValueError(
            "All rows contain errors. No valid data to process.\n" + "\n".join(errors)
        )

    return comments, labels, errors


def convert_label_to_sentiment(labels: list, label_map=None):
    """
    Convert a list of numeric labels (-1, 0, 1) to sentiment strings.
    label_map must be like {"negative": -1, "neutral": 0, "positive": 1}
    """
    if label_map is None:
        label_map = {"negative": -1, "neutral": 0, "positive": 1}

    # Invert the map so we can look up by numeric value
    inverse_map = {v: k for k, v in label_map.items()}

    sentiments = []
    for label in labels:
        try:
            sentiments.append(inverse_map[int(label)])
        except (KeyError, ValueError):
            raise ValueError(f"Label {label} is invalid. Allowed labels: -1, 0, 1.")

    return sentiments


def save_comments_to_csv(comments, labels, output_path="output.csv"):
    """
    Save comments and labels into a CSV file with 2 columns:
        - comment
        - label

    Args:
        comments (list[str]): List of text comments
        labels (list[str]): List of labels (same length as comments)
        output_path (str): Path to save the CSV
    """

    if len(comments) != len(labels):
        raise ValueError("The number of comments and labels must be the same.")

    # Create DataFrame
    df = pd.DataFrame({"comment": comments, "label": labels})

    # Save to CSV
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"CSV successfully saved to {output_path}")


def get_embedder(model_name=EMBEDDER_MODEL):
    logger.info("Loading embedder model: {model_name}")
    embedder = SentenceTransformer(model_name)
    return embedder


def get_test_data(df: pd.DataFrame = None):
    if df is None:
        df = read_data(TEST_DATA_PATH)

    comments, labels, errors = process_data(df)
    return comments, labels, errors


def generate_school_years():
    current_year = datetime.now().year
    school_years = [f"{y}-{y+1}" for y in range(current_year, 2019, -1)]
    return school_years


def generate_semesters():
    semesters = ["1st", "2nd", "Summer"]
    return semesters


def create_job():
    """
    Create a new job entry in JOBS and return the job_id.
    """

    job_id = str(uuid4())
    JOBS[job_id] = {
        "status": "queued",
        "progress": "",
        "message": "Job created",
        "accuracy": "",
        "errors": "",
        "report": "",
        "feedback": "",
    }
    return job_id


def update_job(job_id, **kwargs):
    """
    Update fields of JOBS[job_id] safely.

    Example:
        update_job(job_id, progress=50, message="Halfway done")
    """
    if job_id not in JOBS:
        raise KeyError(f"Job ID '{job_id}' does not exist in JOBS.")

    for key, value in kwargs.items():
        if key in JOBS[job_id]:
            JOBS[job_id][key] = value
        else:
            # optional: allow dynamic fields OR raise an error
            JOBS[job_id][key] = value  # add new fields if needed


def remove_job(job_id):
    """
    Remove a job entry from JOBS.
    """
    if job_id in JOBS:
        del JOBS[job_id]


def get_model(model_name: str):

    model_path = os.path.join(os.getenv("MODEL_DIR"), model_name)
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
    else:
        clf = create_SGD_classifier()
    return clf
