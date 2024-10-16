from pydantic import BaseModel
from enum import Enum


class Question(BaseModel):
    question: str


class RetrieverApproachEnum(str, Enum):
    EMBEDDING_BASED = "embedding-based"
    KEYWORD_MATCH = "keyword-match"
    KNOWLEDGE_GRAPH = "knowledge-graph"


class RetrieverApproach(BaseModel):
    approach: RetrieverApproachEnum
