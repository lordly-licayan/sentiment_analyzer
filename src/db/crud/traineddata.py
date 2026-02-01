from sqlalchemy import desc
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from .. import models, schemas


def create_trained_data(db: Session, data: list[models.TrainedData]):
    try:
        db.bulk_save_objects(data)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise


def list_all_trained_data(db: Session):
    return db.query(models.TrainedData).distinct(models.TrainedData.comment).all()


def list_last_trained_data(db: Session, limit: int = 100):
    return (
        db.query(models.TrainedData)
        .distinct(models.TrainedData.comment)
        .order_by(
            models.TrainedData.comment,
            desc(models.TrainedData.id),
        )
        .limit(limit)
        .all()
    )


def list_trained_data_by_file(db: Session, file_id: str):
    return (
        db.query(models.TrainedData).filter(models.TrainedData.file_id == file_id).all()
    )


def get_trained_data(db: Session, trained_data_id: int):
    return (
        db.query(models.TrainedData)
        .filter(models.TrainedData.id == trained_data_id)
        .first()
    )


def update_trained_data(
    db: Session, trained_data_id: int, update: schemas.TrainedDataCreate
):
    db_trained_data = get_trained_data(db, trained_data_id)
    if not db_trained_data:
        return None

    for k, v in update.dict().items():
        setattr(db_trained_data, k, v)

    db.commit()
    db.refresh(db_trained_data)
    return db_trained_data


def delete_trained_data(db: Session, trained_data_id: int):
    db_trained_data = get_trained_data(db, trained_data_id)
    if not db_trained_data:
        return None

    db.delete(db_trained_data)
    db.commit()
    return True


def paginate_trained_data(
    db: Session, file_id: str = None, limit: int = 100, cursor: int = None
):
    if file_id:
        query = db.query(models.TrainedData).filter(
            models.TrainedData.file_id == file_id
        )
    else:
        query = db.query(models.TrainedData)

    if cursor:
        query = query.filter(models.TrainedData.id < cursor)

    data = query.order_by(desc(models.TrainedData.id)).limit(limit).all()

    # Convert ORM objects to dict
    result = [
        {"id": c.id, "comment": c.comment, "label": c.label, "remarks": c.remarks}
        for c in data
    ]

    return result
