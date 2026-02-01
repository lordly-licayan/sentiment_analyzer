from io import StringIO
import os
from pathlib import Path
from typing import Optional
from fastapi.concurrency import run_in_threadpool
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
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sqlalchemy.orm import Session

import uvicorn
from model.pydantic_model import SentimentRequest, SentimentResponse
from src import (
    ALLOWED_FRAME_ANCESTORS,
    DEFAULT_CLASSIFIER,
    DEFAULT_TRAINED_MODEL_NAME,
    SUPPORTED_CLASSIFIERS,
)
from src.db.crud.traineddata import (
    paginate_trained_data,
)
from src.db.crud.fileinfo import delete_fileinfo, get_fileinfo
from src.db.crud.trainedmodel import (
    delete_trained_model,
    get_trained_model,
)
from src.db.crud.trainedmodelresult import paginate_trained_model_results
from src.db.database import get_db
from src.db.schemas import TrainModelForm, TrainModelFormDependency
from src.helper import (
    create_job,
    get_file_hash,
    get_list_of_trained_models,
    list_of_uploaded_files,
    logger,
    process_payload,
    remove_trained_model,
    retrieve_trained_model,
)
import src.helper as helper

from src.trainer import JOBS, run_trainer

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


@app.middleware("http")
async def iframe_middleware(request: Request, call_next):
    response: Response = await call_next(request)

    allowed_urls = " ".join(u.strip() for u in ALLOWED_FRAME_ANCESTORS.split(","))

    # Allow embedding ONLY from your Flask app domain
    response.headers["Content-Security-Policy"] = f"frame-ancestors {allowed_urls}"

    # Remove X-Frame-Options if present
    if "x-frame-options" in response.headers:
        del response.headers["x-frame-options"]

    return response


# -----------------------------------------------------------
# ROUTES
# -----------------------------------------------------------
@app.post("/train_model")
async def train_model(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    form_data: TrainModelForm = Depends(TrainModelFormDependency),
):
    """Endpoint to handle model training requests.
    Args:
        background (BackgroundTasks): FastAPI background tasks manager.
        file (UploadFile): Uploaded CSV file containing training data.
        form_data (TrainModelForm): Additional training configuration data.
        Returns:
        dict: Contains the job ID for tracking the training process.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV")

    job_id = create_job()

    try:
        content = await file.read()
        file_id = get_file_hash(content)
        df = pd.read_csv(StringIO(content.decode("utf-8", errors="ignore")))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading uploaded file: {e}")

    background.add_task(
        run_trainer, job_id, file_id, file.filename, df, form_data.model_dump()
    )

    return {"job_id": job_id}


@app.get("/training-status/{job_id}")
async def job_status(job_id: str):
    """Endpoint to check the status of a training job by its ID.
    Args:
        job_id (str): The ID of the training job.
        Returns:
        dict: Status information of the job.
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")

    return JOBS[job_id]


@app.post("/predict-sentiment", response_model=SentimentResponse)
async def predict_sentiments(request: SentimentRequest):
    """Endpoint to predict sentiments using a trained model.
    Args:
        request (SentimentRequest): Pydantic model containing model name and text.
        Returns:
        SentimentResponse: Pydantic model containing prediction results.
    """
    if not request.model_name:
        raise HTTPException(status_code=500, detail=f"No model name.")

    if not request.lines:
        raise HTTPException(status_code=500, detail=f"No lines sent.")

    trained_model = retrieve_trained_model(request.model_name)
    if not trained_model:
        logger.warning(f"Model {request.model_name} not found!")
        raise HTTPException(
            status_code=500, detail=f"Model {request.model_name} not found!"
        )

    result = await run_in_threadpool(process_payload, trained_model, request.lines)
    return SentimentResponse(result)


@app.get("/trained-models")
async def get_latest_models(db: Session = Depends(get_db)):
    """
    Async endpoint to retrieve a list of all trained models from the database.
    """
    result = await run_in_threadpool(get_list_of_trained_models, db)
    return result


@app.delete("/delete-model/{model_id}")
async def delete_model(model_id: int, db: Session = Depends(get_db)):
    """
    Endpoint to delete a trained model from the database.
    """
    trained_model = await run_in_threadpool(get_trained_model, db, model_id)
    if not trained_model:
        raise HTTPException(
            status_code=404, detail=f"Trained model with ID {model_id} not found."
        )

    model_deleted = await run_in_threadpool(delete_trained_model, db, trained_model)
    if not model_deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Trained model with ID {model_id} cannot be deleted.",
        )

    is_deleted = remove_trained_model(trained_model.model_name)

    return {"detail": is_deleted}


@app.get("/uploaded-files")
async def get_uploaded_files(db: Session = Depends(get_db)):
    """Endpoint to retrieve a list of all uploaded files from the database.
    Args:
        db (Session): Database session dependency.
        Returns:
        list: List of uploaded files.
    """
    result = await run_in_threadpool(list_of_uploaded_files, db)
    return result


@app.delete("/delete-file/{file_id}")
async def delete_file(file_id: str, db: Session = Depends(get_db)):
    """
    Async endpoint to delete a file and its associated comments from the database.
    """

    file_deleted = await run_in_threadpool(delete_fileinfo, db, file_id)

    if not file_deleted:
        raise HTTPException(
            status_code=404, detail=f"File with ID {file_id} cannot be deleted."
        )

    return {"detail": f"File with ID {file_id} and its comments have been deleted."}


@app.get("/trained-model-results-paging")
async def get_trained_model_results_paging(
    model_id: Optional[int] = None,
    cursor: int | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):

    if not model_id:
        return {"data": []}

    data = await run_in_threadpool(
        paginate_trained_model_results, db, model_id, limit, cursor
    )
    result = {"data": data}
    return result


@app.get("/comments-paging")
async def get_trained_data_paging(
    file_id: Optional[str] = None,
    cursor: int | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    filename = None
    if file_id:
        fileinfo = get_fileinfo(db, file_id)
        if not fileinfo:
            raise HTTPException(
                status_code=404, detail=f"File with ID {file_id} not found."
            )
        filename = fileinfo.filename

    list_of_trained_data = await run_in_threadpool(
        paginate_trained_data, db, file_id, limit, cursor
    )
    result = {"filename": filename, "comments": list_of_trained_data}
    return result


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    """
    Renders the home page with dynamic data such as school years, semesters,
    default model name, supported classifiers, and trained models.
    """
    school_years = helper.generate_school_years()
    semesters = helper.generate_semesters()
    default_model_name = DEFAULT_TRAINED_MODEL_NAME
    supported_classifiers = list(SUPPORTED_CLASSIFIERS.keys())
    default_classifier = DEFAULT_CLASSIFIER

    models = await run_in_threadpool(get_list_of_trained_models, db)

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
    port = int(os.environ.get("PORT", 8070))
    uvicorn.run(app, host="0.0.0.0", port=port)
