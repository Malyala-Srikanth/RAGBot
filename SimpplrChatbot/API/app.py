import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from Data.query_helper import QueryHelper
from Utils.validation import Question


@asynccontextmanager
async def lifespan(app: FastAPI):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Initializing query helper
    query_helper = QueryHelper()
    app.state.query_helper = query_helper

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


@app.get("/health-check")
async def health_check():
    return {"status": "healthy"}


@app.post("/query")
async def query(question: Question):
    try:
        result = app.state.query_helper.query(question)
        answer = result["result"]
        sources = [doc.page_content for doc in result["source_documents"]]

        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(session_id: str, new_query: Question):
    try:
        # Process the new query
        result = app.state.query_helper.chat(new_query, session_id)
        answer = result["result"]
        sources = [doc.page_content for doc in result["source_documents"]]

        return {
            "session_id": session_id,
            "answer": answer,
            "sources": sources,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
