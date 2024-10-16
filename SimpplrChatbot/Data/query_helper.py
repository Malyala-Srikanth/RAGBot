from langchain.chains import (
    RetrievalQA,
    ConversationChain,
)  # Add ConversationChain import
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory  # Add this import

from Data.document_loader import DocumentLoader
from Data.embedding_based_retriever import EmbeddingBasedRetriever
from Data.keyword_match_retriever import KeywordMatchRetriever
from Data.knowledge_graph_retriever import GraphRAG
from API.settings import settings
from Utils.utils import logger

conversations = {}


class QueryHelper:
    def __init__(self):
        self.index_name = settings.INDEX_NAME
        self.approach = settings.RETRIEVER_APPROACH
        self.elasticsearch_url = settings.ELASTICSEARCH_URL
        self.es_client = None  # We'll initialize this only if needed for BM25
        self.vector_store = None
        self.qa_chain = None

    async def initialize(self):
        self.document_loader = DocumentLoader(settings.DATA_PATH)
        documents = await self.document_loader.load_and_process()

        self.llm = ChatOpenAI(
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
            self.retriever = self.vector_store.as_retriever()
        elif self.approach == "keyword-match":
            self.keyword_match_retriever = KeywordMatchRetriever(
                self.index_name, self.elasticsearch_url
            )
            await self.keyword_match_retriever.index_documents(documents)
            self.retriever = await self.keyword_match_retriever.get_retriever()
        elif self.approach == "knowledge-graph":
            logger.info("Loading Knowledge Graph" + "*" * 2000)
            self.graph_rag = GraphRAG()
            await self.graph_rag.process_documents(documents)
        else:
            raise NotImplementedError(f"Unsupported approach: {self.approach}")

    async def query(self, query: str):
        print("2", query)
        if self.approach == "knowledge-graph":
            response = await self.graph_rag.query(query=query)
            return response

        logger.info(f"Using {self.approach} retriever")
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
        )
        return await qa_chain.ainvoke({"query": query})

    async def chat(self, message: str, session_id: str):
        # Check if the session already exists
        if session_id not in conversations:
            memory = ConversationBufferMemory()
            conversation = ConversationChain(llm=self.llm, memory=memory)
            conversations[session_id] = conversation
        else:
            conversation = conversations[session_id]

        # Process the incoming message and get a response
        response = await conversation.predict(input=message)
        return response  # Return the response to the user
