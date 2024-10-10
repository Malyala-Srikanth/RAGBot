import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from Data.document_loader import DocumentLoader

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()

# Initialize DocumentLoader and load documents
data_path = os.getenv("DATA_PATH", "./data")
elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
index_name = "simpplr-chatbot"

document_loader = DocumentLoader(data_path, index_name, elasticsearch_url)
vector_store = document_loader.load_split_and_index()

# Initialize language model
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
)


class Question(BaseModel):
    question: str


@app.get("/health-check")
async def health_check():
    return {"status": "healthy"}


@app.post("/query")
async def query(question: Question):
    try:
        result = qa_chain({"query": question.question})
        answer = result["result"]
        sources = [doc.page_content for doc in result["source_documents"]]

        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
