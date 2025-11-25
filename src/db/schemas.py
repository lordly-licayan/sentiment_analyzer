from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Optional
from fastapi import Form


# ---------------------
# FileInfo Schemas
# ---------------------
class FileInfoBase(BaseModel):
    file_id: str
    filename: str
    no_of_data: int
    date_uploaded: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    remarks: Optional[str] = None


class FileInfoCreate(FileInfoBase):
    pass


class FileInfo(FileInfoBase):
    id: int

    model_config = {"from_attributes": True}


# ---------------------
# Comments Schemas
# ---------------------
class CommentBase(BaseModel):
    file_id: str
    comment: str
    label: int
    remarks: Optional[str] = None


class CommentCreate(CommentBase):
    pass


class Comment(CommentBase):
    id: int

    model_config = {"from_attributes": True}


# ---------------------
# TrainedModel Schemas
# ---------------------
class TrainedModelBase(BaseModel):
    sy: str
    semester: str
    model_name: str
    classifier: str
    accuracy: float
    no_of_data: int
    date_trained: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    remarks: Optional[str] = None


class TrainedModelCreate(TrainedModelBase):
    pass


class TrainedModel(TrainedModelBase):
    id: int

    model_config = {"from_attributes": True}


class TrainModelForm(BaseModel):
    modelName: str
    sy: str
    semester: str
    classifierModel: str


def TrainModelFormDependency(
    modelName: str = Form(...),
    sy: str = Form(...),
    semester: str = Form(...),
    classifierModel: str = Form(...),
):
    return TrainModelForm(
        modelName=modelName, sy=sy, semester=semester, classifierModel=classifierModel
    )
