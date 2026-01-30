from sqlalchemy import desc
from sqlalchemy.orm import Session
from .. import models, schemas

from typing import List, Optional


def create_trained_model_result(db: Session, data: schemas.TrainedModelResultCreate):
    db_model = models.TrainedModelResult(**data.dict())
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model.id


def create_trained_model_results(
    db: Session, data: list[schemas.TrainedModelResultCreate]
):
    db.bulk_insert_mappings(models.TrainedModelResult, [item.dict() for item in data])
    db.commit()


def save_trained_model_results(db: Session, trained_model_id: int, data: list):
    results = [{**item, "trained_model_id": trained_model_id} for item in data]

    db.bulk_insert_mappings(models.TrainedModelResult, results)
    db.commit()


def paginate_results(db: Session, id: int = None, limit: int = 100, cursor: int = None):
    if id:
        query = db.query(models.TrainedModelResult).filter(
            models.TrainedModelResult.id == id
        )
    else:
        query = db.query(models.TrainedModelResult)

    if cursor:
        query = query.filter(models.TrainedModelResult.id < cursor)

    results = query.order_by(desc(models.TrainedModelResult.id)).limit(limit).all()

    # Convert ORM objects to dict
    result = [
        {
            "id": c.id,
            "comment": c.comment,
            "actual_label": c.actual_label,
            "predicted_label": c.predicted_label,
            "confidence": c.confidence,
            "is_matched": c.is_matched,
        }
        for c in results
    ]

    return result
