import os
import re
import time
import traceback
from io import StringIO
from uuid import uuid4

from fastapi.params import Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import uvicorn
from src.helper import logger
import src.helper as helper

from src.trainer import process_data_and_train, JOBS, MODEL_PATH


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
@app.post("/upload")
async def upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    sy: str = Form(...),
    semester: str = Form(...),
    datasetType: str = Form(...),
):
    logger.info(f"Received file upload: {file.filename}")
    logger.info(f"sy: {sy}")
    logger.info(f"semester: {semester}")
    logger.info(f"datasetType: {datasetType}")

    if not file.filename.endswith(".csv"):
        logger.warning("File rejected — not CSV")
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV")

    content = await file.read()

    job_id = str(uuid4())
    JOBS[job_id] = {"status": "queued", "detail": None, "result": None}

    logger.info(f"Created job {job_id} — queued for background processing")
    background.add_task(process_data_and_train, job_id, content)

    return {"job_id": job_id, "status": "queued"}


@app.get("/job-status/{job_id}")
def job_status(job_id: str):
    logger.info(f"Checking job status for {job_id}")

    if job_id not in JOBS:
        logger.warning(f"Job ID not found: {job_id}")
        raise HTTPException(status_code=404, detail="Job ID not found")

    return JOBS[job_id]


@app.post("/predict")
async def predict_text(payload: dict):
    text = payload.get("comment") if isinstance(payload, dict) else None

    if not text:
        logger.warning("Prediction request with no comment")
        raise HTTPException(status_code=400, detail="No comment provided")

    logger.info("Prediction requested")

    if not os.path.exists(MODEL_PATH):
        logger.warning("Prediction failed — model missing")
        raise HTTPException(
            status_code=404,
            detail="Model not trained yet. Upload CSV to train.",
        )

    data = joblib.load(MODEL_PATH)
    clf = data["clf"]

    embedder = helper.get_embedder()
    emb = embedder.encode([text])

    pred = clf.predict(emb)[0]
    probs = (
        clf.predict_proba(emb)[0].tolist() if hasattr(clf, "predict_proba") else None
    )

    logger.info(f"Prediction complete — label={pred}")

    return {"comment": text, "prediction": pred, "probs": probs}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    logger.info("Health check")
    school_years = helper.generate_school_years()
    semesters = helper.generate_semesters()
    dataset_type = helper.generate_dataset_type()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "school_years": school_years,
            "semesters": semesters,
            "dataset_type": dataset_type,
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
