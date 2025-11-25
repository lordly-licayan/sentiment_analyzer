# crud/comments.py
from sqlalchemy.orm import Session
from .. import models, schemas


def create_comment(db: Session, comment: schemas.CommentCreate):
    db_comment = models.Comments(**comment.dict())
    db.add(db_comment)
    db.commit()
    db.refresh(db_comment)
    return db_comment


def create_comments(db: Session, comments: list[schemas.CommentCreate]):
    db.bulk_insert_mappings(models.Comments, [comment.dict() for comment in comments])
    db.commit()


def list_all_comments(db: Session):
    return db.query(models.Comments).distinct(models.Comments.comment).all()


def list_comments_by_file(db: Session, file_id: str):
    return db.query(models.Comments).filter(models.Comments.file_id == file_id).all()


def get_comment(db: Session, comment_id: int):
    return db.query(models.Comments).filter(models.Comments.id == comment_id).first()


def update_comment(db: Session, comment_id: int, update: schemas.CommentCreate):
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
