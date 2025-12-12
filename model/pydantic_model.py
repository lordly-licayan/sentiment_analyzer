from pydantic import BaseModel, RootModel
from typing import Dict


class SentimentRequest(BaseModel):
    model_name: str
    lines: list[str]


class CommentCategory(RootModel[Dict[str, float]]):
    """
    Dynamic dictionary for categories and their scores.
    """


class CommentResult(BaseModel):
    sentiment: str
    top_category: str
    top_category_score: float
    category: CommentCategory


class SentimentResponse(RootModel[Dict[str, CommentResult]]):
    """
    Top-level dictionary mapping comment string -> CommentResult
    """
