from io import StringIO
import os
from pathlib import Path
import shutil
from typing import Optional
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sqlalchemy.orm import Session

import uvicorn
from src import (
    DEFAULT_CLASSIFIER,
    DEFAULT_TRAINED_MODEL_NAME,
    SUPPORTED_CLASSIFIERS,
)
from src.db.crud.comments import list_all_comments, list_comments_by_file
from src.db.crud.fileinfo import list_fileinfo
from src.db.crud.trainedmodel import list_trained_models
from src.db.database import get_db
from src.db.schemas import TrainModelForm, TrainModelFormDependency
from src.helper import (
    create_job,
    get_file_hash,
    get_trained_model,
    logger,
    process_payload,
)
import src.helper as helper

from src.trainer import process_data_and_train, JOBS, run_trainer

BASE_DIR = Path(__file__).resolve().parent

# -----------------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------------
app = FastAPI(title="Sentiment API")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------
# ROUTES
# -----------------------------------------------------------
@app.post("/train_model")
async def train_model(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    form_data: TrainModelForm = Depends(TrainModelFormDependency),
):
    """Endpoint to upload CSV file and start model training in background.
    Validates file type, creates a job, and schedules background task.

    Args:
        background (BackgroundTasks): FastAPI background task manager.
        file (UploadFile): Uploaded CSV file.
        form_data (TrainModelForm): Form data for training parameters.
        db (Session): Database session dependency.
        Returns:
        dict: Response containing job ID.
    """
    if not file.filename.endswith(".csv"):
        logger.warning("File rejected — not CSV")
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV")

    job_id = create_job()
    logger.info(f"Created job {job_id} — scheduling background task")

    try:
        file_content = await file.read()
        file_id = get_file_hash(file_content)
        s = file_content.decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(s))
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading uploaded file: {e}")

    # Convert form data into a safe-to-pass dict (not Pydantic object)
    data = form_data.model_dump()
    print(data)

    background.add_task(run_trainer, job_id, file_id, file.filename, df, data)

    return {"job_id": job_id}


@app.get("/training-status/{job_id}")
def job_status(job_id: str):
    """Endpoint to check the status of a training job by its ID.
    Args:
        job_id (str): The ID of the training job.
        Returns:
        dict: Status information of the job.
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")

    return JOBS[job_id]


@app.post("/predict-sentiment")
async def predict_sentiments(model_name: str, payload: dict):
    """Endpoint to predict sentiments using a specified trained model.
    Args:
        model_name (str): Name of the trained model to use.
        payload (dict): Input data for prediction.
        Returns:
        dict: Prediction results.
    """

    if not model_name:
        raise HTTPException(status_code=500, detail=f"No model name.")

    if not payload:
        raise HTTPException(status_code=500, detail=f"No payload sent.")

    trained_model = get_trained_model(model_name)
    if not trained_model:
        logger.warning(f"Model {model_name} not found!")
        raise HTTPException(status_code=500, detail=f"Model {model_name} not found!")

    result = process_payload(trained_model, payload)
    return result


@app.get("/trained-models")
def get_latest_models(db: Session = Depends(get_db)):
    """Endpoint to retrieve a list of all trained models from the database.
    trained_models = list_ trained_models(db)
    Args:
        db (Session): Database session dependency.
        Returns:
        list: List of trained models.
    """
    trained_models = list_trained_models(db)
    result = [m.to_dict() for m in trained_models]
    return result


@app.get("/uploaded-files")
def get_uploaded_files(db: Session = Depends(get_db)):
    """Endpoint to retrieve a list of all uploaded files from the database.
    Args:
        db (Session): Database session dependency.
        Returns:
        list: List of uploaded files.
    """
    uploaded_files = list_fileinfo(db)
    result = [m.to_dict() for m in uploaded_files]
    return result


@app.get("/comments")
def get_comments(file_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Endpoint to retrieve comments, optionally filtered by file ID.
    Args:
        file_id (Optional[str]): ID of the file to filter comments.
        db (Session): Database session dependency.
        Returns:
        list: List of comments.
    """
    if file_id:
        comments = list_comments_by_file(db, file_id)
    else:
        comments = list_all_comments(db)

    result = [m.to_dict() for m in comments]
    return result


@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    """Renders the home page with dynamic data such as school years, semesters,
    default model name, supported classifiers, and trained models.
    Args:
        request (Request): FastAPI request object.
        db (Session): Database session dependency.
        Returns:
        HTMLResponse: Rendered HTML page.
    """
    school_years = helper.generate_school_years()
    semesters = helper.generate_semesters()
    default_model_name = DEFAULT_TRAINED_MODEL_NAME
    supported_classifiers = list(SUPPORTED_CLASSIFIERS.keys())
    default_classifier = DEFAULT_CLASSIFIER

    trained_models = list_trained_models(db)
    models = [m.to_dict() for m in trained_models]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "school_years": school_years,
            "semesters": semesters,
            "default_model_name": default_model_name,
            "supported_classifiers": supported_classifiers,
            "default_classifier": default_classifier,
            "models": models,
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
