import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from Data.document_loader import DocumentLoader
from API.settings import settings

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)

    # Startup code here
    document_loader = DocumentLoader(
        settings.DATA_PATH,
        settings.INDEX_NAME,
        settings.ELASTICSEARCH_URL,
        embeddings,  # noqa: E501
    )
    vector_store = await document_loader.load_split_and_index()

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

    # Add qa_chain to app state
    app.state.qa_chain = qa_chain

    yield
    # Shutdown code here (if needed)


app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


@app.get("/health-check")
async def health_check():
    return {"status": "healthy"}


@app.post("/query")
async def query(question: Question):
    try:
        result = app.state.qa_chain({"query": question.question})
        answer = result["result"]
        sources = [doc.page_content for doc in result["source_documents"]]

        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
