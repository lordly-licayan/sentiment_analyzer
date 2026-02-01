from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from fastapi import Form


# ---------------------
# FileInfo Schemas
# ---------------------
class FileInfoBase(BaseModel):
    file_id: str
    filename: str
    data_count: int
    date_uploaded: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    remarks: Optional[str] = None


class FileInfoCreate(FileInfoBase):
    pass


class FileInfo(FileInfoBase):
    id: int

    model_config = {"from_attributes": True}


# ---------------------
# TrainedData Schemas
# ---------------------
class TrainedDataBase(BaseModel):
    file_id: str
    comment: str
    label: int
    remarks: Optional[str] = None


class TrainedDataCreate(TrainedDataBase):
    pass


class TrainedData(TrainedDataBase):
    id: int

    model_config = {"from_attributes": True}


# ---------------------
# TrainedModel Schemas
# ---------------------
class TrainedModelBase(BaseModel):
    school_year: str
    semester: str
    model_name: str
    classifier_name: str
    metrics: Dict[str, Any]
    data_count: int
    date_trained: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    remarks: Optional[str] = None


class TrainedModelCreate(TrainedModelBase):
    pass


class TrainedModel(TrainedModelBase):
    id: int

    model_config = {"from_attributes": True}


# ---------------------
# TrainedModelResult Schemas
# ---------------------
class TrainedModelResultBase(BaseModel):
    trained_model_id: int
    comment: str
    actual_label: str
    predicted_label: str
    confidence: Optional[float] = Field(
        None, ge=0, le=100, description="Confidence score (0–100)"
    )
    is_matched: Optional[bool] = None


class TrainedModelResultCreate(TrainedModelResultBase):
    pass


class TrainedModelResult(TrainedModelResultBase):
    id: int

    model_config = {"from_attributes": True}


class TrainModelForm(BaseModel):
    modelName: str
    school_year: str
    semester: str
    classifierModel: str


def TrainModelFormDependency(
    modelName: str = Form(...),
    school_year: str = Form(...),
    semester: str = Form(...),
    classifierModel: str = Form(...),
):
    return TrainModelForm(
        modelName=modelName,
        school_year=school_year,
        semester=semester,
        classifierModel=classifierModel,
    )
