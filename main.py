import os
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
from sqlalchemy.orm import Session

import uvicorn
from src import (
    DEFAULT_CLASSIFIER,
    DEFAULT_TRAINED_MODEL_NAME,
    LABEL_MAP,
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
    get_sentiments,
    get_trained_model,
    logger,
)
import src.helper as helper

from src.trainer import process_data_and_train, JOBS


# -----------------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------------
app = FastAPI(title="Sentiment API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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
    db: Session = Depends(get_db),
):
    try:
        # Validate file extension
        if not file.filename.endswith(".csv"):
            logger.warning("File rejected — not CSV")
            raise HTTPException(status_code=400, detail="Uploaded file must be a CSV")

        filename = file.filename

        try:
            file_content = await file.read()
        except Exception as e:
            logger.error(f"Failed to read uploaded file: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error reading uploaded file: {e}"
            )

        job_id = create_job()
        logger.info(f"Created job {job_id} — queued for background processing")

        background.add_task(
            process_data_and_train, job_id, filename, file_content, form_data, db
        )

        return {"job_id": job_id}

    except Exception as e:
        # Catch-all fallback to avoid unhandled errors
        logger.error(f"Unexpected train_model error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected server error occurred")


@app.get("/training-status/{job_id}")
def job_status(job_id: str):
    logger.info(f"Checking job status for {job_id}")

    if job_id not in JOBS:
        logger.warning(f"Job ID not found: {job_id}")
        raise HTTPException(status_code=404, detail="Job ID not found")

    return JOBS[job_id]


@app.post("/predict-sentiment")
async def predict_sentiments(model_name: str, payload: dict):
    if not model_name:
        raise HTTPException(status_code=500, detail=f"No model name.")

    if not payload:
        raise HTTPException(status_code=500, detail=f"No payload sent.")

    trained_model = get_trained_model(model_name)
    if not trained_model:
        logger.warning(f"Model {model_name} not found!")
        raise HTTPException(status_code=500, detail=f"Model {model_name} not found!")

    sentiments = get_sentiments(trained_model, payload)
    return sentiments


@app.get("/trained-models")
def get_latest_models(db: Session = Depends(get_db)):
    trained_models = list_trained_models(db)
    result = [m.to_dict() for m in trained_models]
    return result


@app.get("/uploaded-files")
def get_uploaded_files(db: Session = Depends(get_db)):
    uploaded_files = list_fileinfo(db)
    result = [m.to_dict() for m in uploaded_files]
    return result


@app.get("/comments")
def get_comments(file_id: Optional[str] = None, db: Session = Depends(get_db)):
    if file_id:
        comments = list_comments_by_file(db, file_id)
    else:
        comments = list_all_comments(db)

    result = [m.to_dict() for m in comments]
    return result


@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
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
