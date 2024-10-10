import re
import asyncio
from typing import List
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # noqa: E501
from langchain_experimental.text_splitter import SemanticChunker

from langchain.vectorstores import ElasticVectorSearch
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv


class DocumentLoader:
    def __init__(
        self, data_path: str, index_name: str, elasticsearch_url: str
    ):  # noqa: E501
        load_dotenv()
        self.data_path = data_path
        self.index_name = index_name
        self.elasticsearch_url = elasticsearch_url
        self.embeddings = OpenAIEmbeddings()

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
        self, documents: List[str]
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
        documents: List[str],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,  # noqa: E501
        type_of_splitter: str = "recursive",
    ) -> List[str]:
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

    async def index_documents(
        self, documents: List[str]
    ) -> ElasticVectorSearch:  # noqa: E501
        vector_store = await asyncio.to_thread(
            ElasticVectorSearch.from_documents,
            documents,
            self.embeddings,
            elasticsearch_url=self.elasticsearch_url,
            index_name=self.index_name,
        )
        return vector_store

    async def load_split_and_index(
        self, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> ElasticVectorSearch:
        print("Loading documents")
        documents = await self.load_documents()
        print("Restructuring documents")
        restructured_docs = await self.restructure_documents(documents)
        print("Splitting documents")
        split_docs = await self.split_documents(
            restructured_docs, chunk_size, chunk_overlap
        )
        for doc in split_docs:
            print(doc.page_content)
            print("*" * 100)
        print("Indexing documents")
        vector_store = await self.index_documents(split_docs)
        return vector_store


if __name__ == "__main__":
    data_path = "/Users/malyala/Desktop/SimpplrChatbot/SimpplrChatbot/pdfs"
    document_loader = DocumentLoader(
        data_path, "simpplr-chatbot", "http://localhost:9200"
    )  # noqa: E501
    vector_store = asyncio.run(document_loader.load_split_and_index())
