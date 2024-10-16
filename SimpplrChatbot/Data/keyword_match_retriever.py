import asyncio
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ElasticsearchBM25Retriever
from Utils.utils import logging


class KeywordMatchRetriever:
    def __init__(self, index_name: str, elasticsearch_url: str):
        self.index_name = index_name
        self.elasticsearch_url = elasticsearch_url
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.retriever = None

    async def split_documents(self, documents: List[Document]) -> List[Document]:
        return await asyncio.to_thread(self.text_splitter.split_documents, documents)

    async def index_documents(self, documents: List[Document]) -> None:
        logging.info("Splitting documents")
        split_docs = await self.split_documents(documents)
        logging.info(f"Split into {len(split_docs)} chunks")

        logging.info("Indexing documents with ElasticsearchBM25Retriever")
        self.retriever = ElasticsearchBM25Retriever(
            elasticsearch_url=self.elasticsearch_url, index_name=self.index_name, k=5
        )
        await asyncio.to_thread(self.retriever.add_documents, split_docs)
        logging.info("Indexing complete")

    async def retrieve(self, query: str, k: int = 5) -> List[Document]:
        if self.retriever is None:
            raise ValueError(
                "Documents have not been indexed yet. Call index_documents() first."
            )

        self.retriever.k = k
        results = await asyncio.to_thread(self.retriever.get_relevant_documents, query)
        return results

    async def get_retriever(self):
        if self.retriever is None:
            raise ValueError(
                "Documents have not been indexed yet. Call index_documents() first."
            )
        return self.retriever


if __name__ == "__main__":
    # Example usage
    index_name = "simpplr_docs"
    elasticsearch_url = "http://localhost:9200"
    retriever = KeywordMatchRetriever(index_name, elasticsearch_url)

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
