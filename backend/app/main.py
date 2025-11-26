from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from pathlib import Path
import os, asyncio
import logging
import asyncio
import os
import chainlit as cl
from backend.rag.pipeline import build_index_from_pdf
from backend.rag.reranker import rerank_results
from backend.rag.refine import llm_refine,chat_with_llm
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from backend.rag.users import validate_user
from chainlit.types import ThreadDict
from dotenv import load_dotenv
import uuid
import asyncio
import logging
from pathlib import Path
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict

load_dotenv()

app = FastAPI()
logger = logging.getLogger(__name__)

# In-memory session store for demo purposes
SESSION = {
    "index": None,
    "index_processing": False,
    "chat_history": [],
}


# Create FastAPI app
app = FastAPI()

# Allow frontend (React) to connect
origins = [
    "http://localhost:8080",   # React dev server default
    "http://localhost:8081",   # Vite auto-switched port
    "http://127.0.0.1:8080",
]
print("succesfully added cors")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # Only allow your frontend
    allow_credentials=True,
    allow_methods=["*"],          # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],          # Allow all headers
)
print("succesfully added cors")
# ---------- Example Routes ----------

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    groq_api_key = os.getenv("GROQ_API_KEY")
    # groq_api_key = 
    if not groq_api_key:
        return {"status": "error", "message": "❌ Error: GROQ_API_KEY not set."}

    user_input = request.query.strip()
    if not user_input:
        return {"status": "error", "message": "Please enter text or upload a PDF to start."}

    index = SESSION.get("index")

    if index:
        retriever = index.as_retriever(similarity_top_k=10)
        results = await asyncio.to_thread(retriever.retrieve, user_input)
        ranked_nodes = await asyncio.to_thread(rerank_results, results, user_input)
        answer = await asyncio.to_thread(llm_refine, ranked_nodes, user_input, groq_api_key)
    else:
        answer = await asyncio.to_thread(chat_with_llm, user_input, groq_api_key)

    # Maintain chat history
    SESSION["chat_history"].append({"user": user_input, "bot": answer})

    return {"status": "success", "answer": answer}




@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if SESSION["index_processing"]:
        return {"status": "error", "message": "⚠️ Indexing already in progress. Please wait."}

    SESSION["index_processing"] = True

    try:
        # Save uploaded file temporarily
        pdf_path = f"temp_{file.filename}"
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        # Build index in background
        index = await asyncio.to_thread(build_index_from_pdf, pdf_path)
        SESSION["index"] = index

        return {"status": "success", "message": "✅ PDF processed! You can now ask questions."}

    except Exception as e:
        logger.exception("Failed to build index")
        return {"status": "error", "message": f"⚠️ Error while processing PDF: {e}"}

    finally:
        SESSION["index_processing"] = False


