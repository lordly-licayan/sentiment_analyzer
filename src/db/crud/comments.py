# crud/comments.py
from sqlalchemy.orm import Session
from .. import models, schemas


def create_comment(db: Session, comment: schemas.CommentsCreate):
    db_comment = models.Comments(**comment.dict())
    db.add(db_comment)
    db.commit()
    db.refresh(db_comment)
    return db_comment


def list_comments_by_file(db: Session, file_id: str):
    return db.query(models.Comments).filter(models.Comments.FileId == file_id).all()


def get_comment(db: Session, comment_id: int):
    return db.query(models.Comments).filter(models.Comments.Id == comment_id).first()


def update_comment(db: Session, comment_id: int, update: schemas.CommentsBase):
    db_comment = get_comment(db, comment_id)
    if not db_comment:
        return None

    for k, v in update.dict().items():
        setattr(db_comment, k, v)

    db.commit()
    db.refresh(db_comment)
    return db_comment


def delete_comment(db: Session, comment_id: int):
    db_comment = get_comment(db, comment_id)
    if not db_comment:
        return None

    db.delete(db_comment)
    db.commit()
    return True
