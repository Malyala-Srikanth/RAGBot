from langchain_core.embeddings import OpenAIEmbeddings
from langchain.vectorstores import ElasticVectorSearch
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import ElasticSearchBM25Retriever

from Data.document_loader import DocumentLoader
from API.settings import settings


class QueryHelper:
    def __init__(self):
        self.index_name = settings.INDEX_NAME
        self.elasticsearch_url = settings.ELASTICSEARCH_URL
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self.vector_store = ElasticVectorSearch(
            index_name=self.index_name,
            embedding_function=self.embeddings,
            elasticsearch_url=self.elasticsearch_url,
        )
        self.document_loader = DocumentLoader(
            data_path=settings.DATA_PATH,
            index_name=self.index_name,
            elasticsearch_url=self.elasticsearch_url,
            embeddings=self.embeddings,
        )
        self.es_client = self.vector_store.client

    def query(self, query: str, approach: str = "embedding-based"):
        if approach == "embedding-based":
            vector_store = self.document_loader.load_split_and_index(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                type_of_splitter=settings.SPLITTER_TYPE,
            )

            # Initialize language model
            llm = ChatOpenAI(
                temperature=settings.LLM_TEMPERATURE,
                top_p=settings.LLM_TOP_P,
                model_name=settings.LLM_MODEL,
            )

            # Initialize qa_chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(),
                return_source_documents=True,
            )

            return qa_chain.invoke({"query": query})
        elif approach == "keyword-match":
            # Initialize BM25 retriever
            bm25_retriever = ElasticSearchBM25Retriever(
                client=self.es_client,
                index_name=self.index_name,
                k=settings.BM25_K_DOCUMENTS,
            )

            # Initialize language model
            llm = ChatOpenAI(
                temperature=settings.LLM_TEMPERATURE,
                top_p=settings.LLM_TOP_P,
                model_name=settings.LLM_MODEL,
            )

            # Initialize qa_chain with BM25 retriever
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=bm25_retriever,
                return_source_documents=True,
            )

            return qa_chain.invoke({"query": query})
        else:
            raise NotImplementedError
