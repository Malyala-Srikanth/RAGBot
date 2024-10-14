import re
import asyncio
from typing import List
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # noqa: E501
from langchain_experimental.text_splitter import SemanticChunker
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings

from langchain.vectorstores import ElasticVectorSearch
from Utils.utils import logging
from tenacity import retry, stop_after_attempt, wait_exponential


class DocumentLoader:
    def __init__(
        self,
        data_path: str,
        index_name: str,
        elasticsearch_url: str,
        embeddings: Embeddings,
    ):  # noqa: E501
        self.data_path = data_path
        self.index_name = index_name
        self.elasticsearch_url = elasticsearch_url
        self.embeddings = embeddings

    async def load_documents(self) -> List[str]:
        # Load text files
        text_loader = DirectoryLoader(
            self.data_path, glob="**/*.txt", loader_cls=TextLoader
        )
        text_documents = await asyncio.to_thread(text_loader.load)

        # Load PDF files
        pdf_loader = DirectoryLoader(
            self.data_path, glob="**/*.pdf", loader_cls=PyPDFLoader
        )
        pdf_documents = await asyncio.to_thread(pdf_loader.load)

        # Combine text and PDF documents
        documents = text_documents + pdf_documents
        return documents

    async def restructure_documents(
        self, documents: List[Document]
    ) -> List[str]:  # noqa: E501
        restructured_documents = []
        for doc in documents:
            page_content = doc.page_content
            metadata = doc.metadata
            # Replace newlines with spaces
            page_content = re.sub(r"\n", " ", page_content)
            # Replace space after a period with a newline, if the period is
            # preceded by a word
            page_content = re.sub(r"(\w{2,}\.)", r"\1\n", page_content)

            restructured_documents.append(
                doc.__class__(page_content=page_content, metadata=metadata)
            )
        return restructured_documents

    async def split_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,  # noqa: E501
        type_of_splitter: str = "recursive",
    ) -> List[Document]:
        if type_of_splitter == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        elif type_of_splitter == "semantic":
            text_splitter = SemanticChunker(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        split_docs = await asyncio.to_thread(
            text_splitter.split_documents, documents
        )  # noqa: E501
        return split_docs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),  # noqa: E501
    )
    async def generate_embeddings(self, texts):
        try:
            embeddings = await asyncio.to_thread(
                self.embeddings.embed_documents, texts
            )  # noqa: E501
            if not embeddings:
                raise ValueError("Embeddings list is empty")
            return embeddings
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise

    async def index_documents(
        self, documents: List[Document]
    ) -> ElasticVectorSearch:  # noqa: E501
        logging.info(f"Indexing {len(documents)} documents")

        if not documents:
            raise ValueError("No documents to index")

        texts = [doc.page_content for doc in documents]

        try:
            embeddings = await self.generate_embeddings(texts)
            logging.info(f"Generated {len(embeddings)} embeddings")

            vector_store = await asyncio.to_thread(
                ElasticVectorSearch.from_documents,
                documents,
                self.embeddings,
                elasticsearch_url=self.elasticsearch_url,
                index_name=self.index_name,
            )
            return vector_store
        except Exception as e:
            logging.error(f"Error during indexing: {str(e)}")
            raise

    async def load_split_and_index(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        type_of_splitter: str = "recursive",
    ) -> ElasticVectorSearch:
        logging.info("Loading documents")
        documents = await self.load_documents()
        logging.info(f"Loaded {len(documents)} documents")
        logging.info("Restructuring documents")
        restructured_docs = await self.restructure_documents(documents)
        logging.info("Splitting documents")
        split_docs = await self.split_documents(
            restructured_docs, chunk_size, chunk_overlap, type_of_splitter
        )
        logging.info("Indexing documents")
        vector_store = await self.index_documents(split_docs)
        return vector_store


if __name__ == "__main__":
    from langchain.embeddings import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    data_path = "/Users/malyala/Desktop/SimpplrChatbot/SimpplrChatbot/pdfs"
    document_loader = DocumentLoader(
        data_path, "simpplr-chatbot", "http://localhost:9200", embeddings
    )  # noqa: E501
    vector_store = asyncio.run(document_loader.load_split_and_index())
