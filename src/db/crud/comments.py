from sqlalchemy import desc
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


def list_last_comments(db: Session, limit: int = 100):
    return (
        db.query(models.Comments)
        .distinct(models.Comments.comment)
        .order_by(
            models.Comments.comment,
            desc(models.Comments.id),
        )
        .limit(limit)
        .all()
    )


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


def paginate_comments(
    db: Session, file_id: str = None, limit: int = 100, cursor: int = None
):
    if file_id:
        query = db.query(models.Comments).filter(models.Comments.file_id == file_id)
    else:
        query = db.query(models.Comments)

    if cursor:
        query = query.filter(models.Comments.id < cursor)

    comments = query.order_by(desc(models.Comments.id)).limit(limit).all()

    # Convert ORM objects to dict
    result = [
        {"id": c.id, "comment": c.comment, "label": c.label, "remarks": c.remarks}
        for c in comments
    ]

    return result
