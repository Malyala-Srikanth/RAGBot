from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from Data.document_loader import DocumentLoader
from Data.embedding_based_retriever import EmbeddingBasedRetriever
from Data.keyword_match_retriever import KeywordMatchRetriever
from API.settings import settings
from Utils.utils import logger
import asyncio


class QueryHelper:
    def __init__(self):
        self.index_name = settings.INDEX_NAME
        self.approach = settings.RETRIEVER_APPROACH
        self.elasticsearch_url = settings.ELASTICSEARCH_URL
        self.es_client = None  # We'll initialize this only if needed for BM25
        self.vector_store = None
        self.qa_chain = None
        # run intialize here
        asyncio.run(self.initialize())

    async def initialize(self):
        self.document_loader = DocumentLoader(settings.DATA_PATH)
        documents = await self.document_loader.load_and_process()

        llm = ChatOpenAI(
            temperature=settings.LLM_TEMPERATURE,
            top_p=settings.LLM_TOP_P,
            model_name=settings.LLM_MODEL,
        )

        if self.approach == "embedding-based":
            self.embedding_based_retriever = EmbeddingBasedRetriever(
                self.index_name, self.elasticsearch_url
            )
            await self.embedding_based_retriever.index_documents(documents)
            self.vector_store = await self.embedding_based_retriever.get_vector_store()
            retriever = self.vector_store.as_retriever()
        elif self.approach == "keyword-match":
            self.keyword_match_retriever = KeywordMatchRetriever(
                self.index_name, self.elasticsearch_url
            )
            await self.keyword_match_retriever.index_documents(documents)
            retriever = self.keyword_match_retriever.get_retriever()

        else:
            raise NotImplementedError(f"Unsupported approach: {self.approach}")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

    async def query(self, query: str):
        logger.info(f"Using {self.approach} retriever")
        return await self.qa_chain.ainvoke({"query": query})
