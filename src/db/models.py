from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, func
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class FileInfo(Base):
    __tablename__ = "fileinfotbl"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String(255), unique=True, index=True)
    filename = Column(String(255), nullable=False)
    no_of_data = Column(Integer, nullable=False, default=0)
    date_uploaded = Column(DateTime(timezone=True), server_default=func.now())
    remarks = Column(Text, nullable=True)

    # Relationship → one file has many comments
    comments = relationship("Comments", back_populates="file", cascade="all, delete")


class Comments(Base):
    __tablename__ = "commentstbl"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String(255), ForeignKey("fileinfotbl.file_id"), nullable=False)
    comment = Column(Text, nullable=False)
    label = Column(Integer, nullable=False)
    remarks = Column(Text, nullable=False)

    # Relationship → comment belongs to file
    file = relationship("FileInfo", back_populates="comments")

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}


class TrainedModel(Base):
    __tablename__ = "trainedmodeltbl"

    id = Column(Integer, primary_key=True, index=True)
    sy = Column(String(50), nullable=False)
    semester = Column(String(50), nullable=False)
    model_name = Column(String(255), nullable=False)
    classifier = Column(String(255), nullable=False)
    accuracy = Column(Float, nullable=True)
    no_of_data = Column(Integer, nullable=False)
    date_trained = Column(DateTime(timezone=True), server_default=func.now())
    remarks = Column(String(255), nullable=True)

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}

        # Format datetime → readable string
        if d.get("date_trained"):
            d["date_trained"] = d["date_trained"].strftime("%Y-%m-%d %H:%M:%S")

        return d
