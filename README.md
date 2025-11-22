Sentiment FastAPI Project
-------------------------

This demo project provides:
- FastAPI backend with endpoints:
  - POST /upload-csv  -> Upload CSV, starts background training job (returns job_id)
  - GET  /job-status/{job_id} -> Poll job status
  - POST /predict -> Predict using trained model (comment in JSON)
- Frontend: simple index.html (upload + spinner + polling)
- Training uses Sentence-Transformers ("paraphrase-multilingual-MiniLM-L12-v2") + LogisticRegression

How to run locally:
1. (Optional) create a venv and activate it
2. pip install -r backend/requirements.txt
3. From project root: uvicorn backend.src.main:app --reload --host 0.0.0.0 --port 8080
4. Open http://localhost:8080/frontend/index.html in your browser and upload a CSV file.

CSV format: must contain 'comments' and 'label' columns.# sentiment_analyzer
