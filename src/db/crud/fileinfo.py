from sqlalchemy.orm import Session
from .. import models, schemas


def create_fileinfo(db: Session, file: schemas.FileInfoCreate):
    db_file = models.FileInfo(**file.dict())
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return db_file


def get_fileinfo(db: Session, file_id: str):
    return db.query(models.FileInfo).filter(models.FileInfo.file_id == file_id).first()


def list_fileinfo(db: Session):
    return (
        db.query(models.FileInfo).order_by(models.FileInfo.date_uploaded.desc()).all()
    )


def update_fileinfo(db: Session, file_id: str, update_data: schemas.FileInfoCreate):
    db_file = get_fileinfo(db, file_id)
    if not db_file:
        return None

    for key, value in update_data.dict().items():
        setattr(db_file, key, value)

    db.commit()
    db.refresh(db_file)
    return db_file


def delete_fileinfo(db: Session, file_id: str):
    db_file = get_fileinfo(db, file_id)
    if not db_file:
        return None

    db.delete(db_file)
    db.commit()
    return True
