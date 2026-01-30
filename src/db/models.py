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

    # one file → many comments
    comments = relationship(
        "Comment", back_populates="file_info", cascade="all, delete-orphan"
    )

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        if d.get("date_uploaded"):
            d["date_uploaded"] = d["date_uploaded"].strftime("%Y-%m-%d %H:%M:%S")
        return d


class Comment(Base):
    __tablename__ = "comment"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(
        Integer, ForeignKey("file_info.id", ondelete="CASCADE"), nullable=False
    )
    comment = Column(Text, nullable=False)
    label = Column(Integer, nullable=False)
    remarks = Column(Text)

    # many comments → one file
    file_info = relationship("FileInfo", back_populates="comments")

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

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        if d.get("date_trained"):
            d["date_trained"] = d["date_trained"].strftime("%Y-%m-%d %H:%M:%S")
        return d
