import os
import re
import asyncio
from typing import List
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.docstore.document import Document
from Utils.utils import logging


class DocumentLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    async def load_documents(self) -> List[Document]:
        # Load PDF files
        pdf_loader = DirectoryLoader(
            self.data_path, glob="**/*.pdf", loader_cls=PyPDFLoader
        )
        pdf_documents = await asyncio.to_thread(pdf_loader.load)
        return pdf_documents

    async def restructure_documents(self, documents: List[Document]) -> List[Document]:
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
                Document(page_content=page_content, metadata=metadata)
            )
        return restructured_documents

    async def load_and_process(self) -> List[Document]:
        logging.info("Loading PDF documents")
        documents = await self.load_documents()
        logging.info(f"Loaded {len(documents)} PDF documents")
        logging.info("Restructuring documents")
        restructured_docs = await self.restructure_documents(documents)
        return restructured_docs


if __name__ == "__main__":
    data_path = os.environ.get("DATA_PATH", "")
    if not data_path:
        raise ValueError("Data path not defined in environment variables")
    document_loader = DocumentLoader(data_path)
    processed_docs = asyncio.run(document_loader.load_and_process())
    print(f"Processed {len(processed_docs)} documents")
