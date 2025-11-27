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



MAX_TURNS = 12       # total raw turns allowed before compression
KEEP_RECENT = 4      # keep last N raw turns visible

# -----------------------------
# Summarizer helper
# -----------------------------
async def summarize_history(history_text: str, groq_api_key: str) -> str:
    """
    Call your LLM summarizer here.
    Replace this stub with your kernel/LLM call.
    """
    # Example: just return first 200 chars as "summary"
    # Replace with your actual LLM invocation
    return "Summary of conversation: " + history_text[:200]



# -----------------------------
# Compression helper
# -----------------------------

async def maybe_compress_history(chat_history, groq_api_key, thread_id=None):

    """
    If chat_history is too long, compress older turns into a single summary entry.
    Keeps session memory small while retaining context.
    """
    
    prev_summary = None
    if (chat_history
        and chat_history[0].get("role") == "system"
        and chat_history[0].get("content", "").startswith("SUMMARY:")):

        ## Take the summary entry from the first slot of chat history, 
        # and strip off the "SUMMARY: " marker so we’re left with only the clean summary text.
        prev_summary = chat_history[0]["content"].replace("SUMMARY: ", "")
        raw_turns = chat_history[1:]
    else:
        raw_turns = chat_history

    if len(raw_turns) <= MAX_TURNS:
        return chat_history  # no compression needed

    # Select old turns to summarize
    older = raw_turns[:-KEEP_RECENT]
    recent = raw_turns[-KEEP_RECENT:]

    # Build text to summarize
    parts = []
    if prev_summary:
        parts.append("Previous summary:\n" + prev_summary)

    # if there is a previous summary, include it in the text to be summarized
    #  otherwise directly parts will run
    parts.extend([f"{t['role']}: {t['content']}" for t in older])
    text_to_summarize = "\n".join(parts)

    # Summarize
    summary = await summarize_history(text_to_summarize, groq_api_key)
    summary = summary.strip().replace("\n", " ")

    summary_entry = {"role": "system", "content": f"SUMMARY: {summary}"}
    new_history = [summary_entry] + recent

    # # Optional: persist summary into thread metadata
    # if thread_id:
    #     await persist_summary_to_thread_metadata(thread_id, summary)

    return new_history


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    print(f"Authenticating user: {username}")
    user = validate_user(username, password)
    if user:
        return cl.User(
            identifier=username,
        )
    return None


@cl.on_chat_start
async def start():
    cl.user_session.set("index", None)
    await cl.Message(content="Welcome! Please upload a PDF.").send()


@cl.data_layer
def get_data_layer():
    """
    Returns the SQLAlchemyDataLayer instance, connecting Chainlit to a PostgreSQL database.
    The DATABASE_URL environment variable is read to get the connection string.
    """
    return SQLAlchemyDataLayer(conninfo=os.getenv("DATABASE_URL"))


# The 'on_chat_resume' function is called when a user resumes a previous chat.
# It rebuilds the chat history from the database to maintain context.
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """
    Handles the resumption of a chat session.
    It retrieves the chat history from the thread steps and stores it in the user session.
    """
    # Start the Ollama session again to ensure the model is ready.
    
    # Initialize the user session's chat history.
    cl.user_session.set("chat_history", [])

    # Loop through the steps (messages) in the saved thread.
    for message in thread['steps']:
        print("Resuming message:", message)  # Debugging line
        # Append the user's messages to the chat history.
        if message['type'] == 'user_message':
            cl.user_session.get("chat_history").append(
                {"role": "user", "content": message['output']}
            )
            print("Current chat history:", cl.user_session.get("chat_history"))  # Debugging line
        # Append the assistant's messages to the chat history.
        elif message['type'] == 'assistant_message':
            cl.user_session.get("chat_history").append(
                {"role": "assistant", "content": message['output']}
            )


logger = logging.getLogger(__name__)


@cl.on_message
async def main(message: cl.Message):
    """
    Robust handler: 
    - If PDF uploaded -> build index
    - If query -> use RAG if index exists, else fallback to normal chat
    - Maintains chat_history for both flows
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        await cl.Message(content="❌ Error: GROQ_API_KEY not set.").send()
        return

    # --- 1) Detect PDF upload ---
    pdf_file_element = None
    if message.elements:
        for el in message.elements:
            path = getattr(el, "path", None)
            if path and Path(path).suffix.lower() == ".pdf":
                pdf_file_element = el
                break

    # --- 2) PDF upload path (indexing) ---
    if pdf_file_element:
        if cl.user_session.get("index_processing"):
            await cl.Message(content="⚠️ Indexing already in progress. Please wait.").send()
            return

        cl.user_session.set("index_processing", True)
        await cl.Message(content="📄 Processing PDF — this may take a moment...").send()

        try:
            index = await asyncio.to_thread(build_index_from_pdf, pdf_file_element.path)
            cl.user_session.set("index", index)
            await cl.Message(content="✅ PDF processed! You can now ask questions about it.").send()
        except Exception as e:
            logger.exception("Failed to build index")
            await cl.Message(content=f"⚠️ Error while processing PDF: {e}").send()
        finally:
            cl.user_session.set("index_processing", False)
        return

    # --- 3) Conversation / Query path ---
    user_input = (message.content or "").strip()
    if not user_input:
        await cl.Message(content="Please enter text or upload a PDF to start.").send()
        return

    index = cl.user_session.get("index")

    try:
        thread_id = cl.user_session.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            cl.user_session.set("thread_id", thread_id)
            
        # --- 3a) Manage chat_history ---
        # context = cl.context.current()
        # thread_id = context.session.thread_id
        chat_history = cl.user_session.get("chat_history", [])
        chat_history.append({"role": "user", "content": message.content})

        # compress long history
        chat_history = await maybe_compress_history(chat_history, groq_api_key, thread_id)
        cl.user_session.set("chat_history", chat_history)

        # --- 3b) Retrieval-Augmented path ---
        if index:
            retriever = index.as_retriever(similarity_top_k=10)
            results = await asyncio.to_thread(retriever.retrieve, user_input)
            ranked_nodes = await asyncio.to_thread(rerank_results, results, user_input)

            # LLM refine with retrieved context
            answer = await asyncio.to_thread(llm_refine, ranked_nodes, user_input, groq_api_key)

        # --- 3c) Fallback pure LLM chat ---
        else:
            answer = await asyncio.to_thread(chat_with_llm, user_input, groq_api_key)

        # --- 3d) Save assistant reply back into history ---
        chat_history.append({"role": "assistant", "content": answer})
        cl.user_session.set("chat_history", chat_history)

        await cl.Message(content=f"**Answer:**\n\n{answer}").send()

    except Exception as e:
        logger.exception("Error handling user query")
        await cl.Message(content=f"⚠️ Error while answering: {e}").send()
