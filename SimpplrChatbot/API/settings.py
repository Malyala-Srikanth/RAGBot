import os

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings
from SimpplrChatbot.Utils.validation import RetrieverApproach, RetrieverApproachEnum
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """
    Settings class to hold all the environment variables
    """

    model_config = ConfigDict(extra="allow")

    LOG_LEVEL: str = "DEBUG"
    PROCESS_POOL_WORKERS: int = 10
    CHAT_HISTORY_LENGTH: int = 10

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "claude-3-5-sonnet-20240620"
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 1.0
    LLM_TOP_K: int = 0
    LLM_MAX_TOKENS: int = 8000

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SPLITTER_TYPE: str = "recursive"
    RETRIEVER_APPROACH: RetrieverApproach = Field(
        default_factory=lambda: RetrieverApproach(
            approach=RetrieverApproachEnum(
                os.getenv(
                    "RETRIEVER_APPROACH", RetrieverApproachEnum.EMBEDDING_BASED.value
                )
            )
        )
    )

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    DATA_PATH: str = os.getenv("DATA_PATH", "./data")
    INDEX_NAME: str = os.getenv("INDEX_NAME", "simpplr-chatbot")
    ELASTICSEARCH_URL: str = os.getenv(
        "ELASTICSEARCH_URL", "http://localhost:9200"
    )  # noqa: E501


settings = Settings()
