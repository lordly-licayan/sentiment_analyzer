from sqlalchemy.orm import Session
from .. import models, schemas


def create_trained_model(db: Session, model: schemas.TrainedModelCreate):
    db_model = models.TrainedModel(**model.dict())
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model.id


def list_trained_models(db: Session):
    return (
        db.query(models.TrainedModel)
        .order_by(models.TrainedModel.date_trained.desc())
        .all()
    )


def get_trained_model(db: Session, model_id: int):
    return (
        db.query(models.TrainedModel).filter(models.TrainedModel.id == model_id).first()
    )


def get_trained_model_name(db: Session, model_name: str):
    return (
        db.query(models.TrainedModel)
        .filter(models.TrainedModel.model_name == model_name)
        .first()
    )


def update_trained_model(
    db: Session, model_id: int, update: schemas.TrainedModelCreate
):
    db_model = get_trained_model(db, model_id)
    if not db_model:
        return None

    for key, val in update.dict().items():
        setattr(db_model, key, val)

    db.commit()
    db.refresh(db_model)
    return db_model


def delete_trained_model(db: Session, model_id: int):
    db_model = get_trained_model(db, model_id)
    if not db_model:
        return None

    db.delete(db_model)
    db.commit()
    return True


def delete_trained_model(db: Session, trained_model: models.TrainedModel):
    db.delete(trained_model)
    db.commit()
    return True
