import asyncio
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import ElasticsearchStore
from Utils.utils import logging
from SimpplrChatbot.API.settings import settings


class EmbeddingBasedRetriever:
    def __init__(self, index_name: str, elasticsearch_url: str):
        self.index_name = index_name
        self.elasticsearch_url = elasticsearch_url
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

    async def split_documents(self, documents: List[Document]) -> List[Document]:
        return await asyncio.to_thread(self.text_splitter.split_documents, documents)

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embeddings.embed_documents, texts)

    async def index_documents(self, documents: List[Document]) -> None:
        logging.info("Splitting documents")
        split_docs = await self.split_documents(documents)
        logging.info(f"Split into {len(split_docs)} chunks")

        logging.info("Generating embeddings")
        texts = [doc.page_content for doc in split_docs]
        embeddings = await self.generate_embeddings(texts)

        logging.info("Indexing documents in Elasticsearch")
        es_store = ElasticsearchStore(
            es_url=self.elasticsearch_url,
            index_name=self.index_name,
            embedding=self.embeddings,
        )
        await asyncio.to_thread(es_store.add_documents, split_docs)

    async def retrieve(self, query: str, k: int = 5) -> List[Document]:
        es_store = ElasticsearchStore(
            es_url=self.elasticsearch_url,
            index_name=self.index_name,
            embedding=self.embeddings,
        )
        results = await asyncio.to_thread(es_store.similarity_search, query, k=k)
        return results

    async def get_vector_store(self):
        if self.vector_store is None:
            embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL, openai_api_key=settings.OPENAI_API_KEY
            )

            self.vector_store = ElasticsearchStore(
                es_url=settings.ELASTICSEARCH_URL,
                index_name=settings.INDEX_NAME,
                embedding=embeddings,
            )

        return self.vector_store


if __name__ == "__main__":
    # Example usage
    index_name = "simpplr_docs"
    elasticsearch_url = "http://localhost:9200"
    retriever = EmbeddingBasedRetriever(index_name, elasticsearch_url)

    # Assuming you have a list of documents from the DocumentLoader
    from document_loader import DocumentLoader

    data_path = "/Users/malyala/Desktop/SimpplrChatbot/SimpplrChatbot/pdfs"
    document_loader = DocumentLoader(data_path)
    documents = asyncio.run(document_loader.load_and_process())

    # Index the documents
    asyncio.run(retriever.index_documents(documents))

    # Retrieve documents
    query = "What is Simpplr's mission?"
    results = asyncio.run(retriever.retrieve(query))
    print(f"Retrieved {len(results)} documents for query: '{query}'")
    for doc in results:
        print(f"- {doc.page_content[:100]}...")
