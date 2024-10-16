from pydantic import BaseModel
from enum import Enum


class Question(BaseModel):
    question: str


class RetrieverApproachEnum(str, Enum):
    EMBEDDING_BASED: str = "embedding-based"
    KEYWORD_MATCH: str = "keyword-match"
    KNOWLEDGE_GRAPH: str = "knowledge-graph"


class RetrieverApproach(BaseModel):
    approach: RetrieverApproachEnum
