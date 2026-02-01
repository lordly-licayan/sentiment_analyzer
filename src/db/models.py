from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Text,
    func,
)
from sqlalchemy.orm import relationship
from .database import Base


from sqlalchemy import (
    JSON,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Text,
    func,
)
from sqlalchemy.orm import relationship
from .database import Base


class FileInfo(Base):
    __tablename__ = "file_info"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String(255), unique=True, index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    data_count = Column(Integer, nullable=False, default=0)
    date_uploaded = Column(DateTime(timezone=True), server_default=func.now())
    remarks = Column(Text)

    # one file → many trained data
    trained_data = relationship(
        "TrainedData", back_populates="file_info", cascade="all, delete-orphan"
    )

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        if d.get("date_uploaded"):
            d["date_uploaded"] = d["date_uploaded"].strftime("%Y-%m-%d %H:%M:%S")
        return d


class TrainedData(Base):
    __tablename__ = "trained_data"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(
        Integer, ForeignKey("file_info.file_id", ondelete="CASCADE"), nullable=False
    )
    comment = Column(Text, nullable=False)
    label = Column(Integer, nullable=False)
    remarks = Column(Text)

    # many trained data → one file
    file_info = relationship("FileInfo", back_populates="trained_data")

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class TrainedModel(Base):
    __tablename__ = "trained_model"

    id = Column(Integer, primary_key=True, index=True)
    school_year = Column(String(50), nullable=False)
    semester = Column(String(50), nullable=False)
    model_name = Column(String(255), nullable=False)
    classifier_name = Column(String(255), nullable=False)
    metrics = Column(JSON, nullable=False, default=dict)
    data_count = Column(Integer, nullable=False, default=0)
    date_trained = Column(DateTime(timezone=True), server_default=func.now())
    remarks = Column(Text)

    results = relationship(
        "TrainedModelResult",
        back_populates="trained_model",
        cascade="all, delete-orphan",
    )

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        if d.get("date_trained"):
            d["date_trained"] = d["date_trained"].strftime("%Y-%m-%d %H:%M:%S")
        return d


class TrainedModelResult(Base):
    __tablename__ = "trained_model_result"

    id = Column(Integer, primary_key=True, index=True)

    trained_model_id = Column(
        Integer,
        ForeignKey("trained_model.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    comment = Column(Text, nullable=False)
    actual_label = Column(String(50), nullable=False)
    predicted_label = Column(String(50), nullable=False)
    confidence = Column(Float)
    is_matched = Column(Boolean, default=False)

    # Relationship → many results belong to one trained model
    trained_model = relationship("TrainedModel", back_populates="results")

    def to_dict(self):
        return {
            "id": self.id,
            "trained_model_id": self.trained_model_id,
            "comment": self.comment,
            "actual_label": self.actual_label,
            "predicted_label": self.predicted_label,
            "confidence": (
                round(self.confidence, 2) if self.confidence is not None else None
            ),
            "is_matched": self.is_matched,
        }
