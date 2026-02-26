from sqlalchemy.orm import Session
from .. import models, schemas


def admin_settings_model(db: Session, model: schemas.AdminSettingsCreate):
    admin_settings = models.AdminSettings(**model.dict())
    db.add(admin_settings)
    db.commit()
    db.refresh(admin_settings)
    return admin_settings.id


def get_admin_settings(db: Session):
    admin_settings = db.query(models.AdminSettings).filter().first()
    categories = []
    if admin_settings and admin_settings.categories:
        categories = [
            category.strip() for category in admin_settings.categories.split(",")
        ]

    return categories
