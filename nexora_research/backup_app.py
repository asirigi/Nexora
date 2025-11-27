import chainlit as cl
import os
import tempfile
import nltk
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter

# --- LlamaIndex and ChromaDB Imports ---
from llama_index.core.schema import Document, NodeWithScore, QueryBundle, TextNode
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import chromadb
from uuid import uuid4
import asyncio
import re
# DataLoader(dataset, batch_size=32, pin_memory=False)


# --- NLTK Download (run once) ---
# This ensures that the 'punkt' tokenizer is available for sentence tokenization.
# It's good practice to put this where it runs only once, e.g., on app startup.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Downloading now...")
    nltk.download('punkt')
    print("NLTK 'punkt' tokenizer downloaded.")


# --- Chunking Functions ---

def chunk_markdown_by_heading(structured_text):
    """
    Chunks markdown text based on H2 headings (##).
    Each chunk starts with an H2 heading and includes all content until the next H2.
    """
    lines = structured_text.strip().split('\n')
    chunks = []
    current_chunk_lines = []
    chunk_number = 1

    for line in lines:
        # Check for H2 heading, but not H3 (to avoid sub-sub-sections breaking main chunks)
        if line.strip().startswith("##") and not line.strip().startswith("###"):
            if current_chunk_lines:
                # If there's content in the current chunk, save it
                content = '\n'.join(current_chunk_lines).strip()
                chunks.append((f"chunk {chunk_number}", content))
                chunk_number += 1
                current_chunk_lines = [] # Reset for the new chunk
        current_chunk_lines.append(line)

    # Add the last chunk if it exists
    if current_chunk_lines:
        content = '\n'.join(current_chunk_lines).strip()
        chunks.append((f"chunk {chunk_number}", content))
    return chunks

def chunk_text_for_embedding(chunk_texts, model_name="BAAI/bge-base-en-v1.5", max_tokens=512, buffer_tokens=20):
    """
    Further chunks text into smaller pieces suitable for embedding models,
    respecting a maximum token limit and sentence boundaries.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_final_chunks = []

    for text in chunk_texts:
        sentences = nltk.sent_tokenize(text)
        current_chunk_sentences = []
        current_tokens = 0

        for sentence in sentences:
            # Calculate token count for the current sentence
            token_count = len(tokenizer.encode(sentence, add_special_tokens=False))

            # If adding the current sentence doesn't exceed the max_tokens (with buffer)
            if current_tokens + token_count <= max_tokens - buffer_tokens:
                current_chunk_sentences.append(sentence)
                current_tokens += token_count
            else:
                # If it exceeds, save the current chunk and start a new one
                if current_chunk_sentences: # Ensure there's content to save
                    all_final_chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentence] # Start new chunk with current sentence
                current_tokens = token_count

        # Add any remaining sentences in the current_chunk_sentences
        if current_chunk_sentences:
            all_final_chunks.append(" ".join(current_chunk_sentences))
            # print(all_final_chunks)
    return all_final_chunks

# --- Index Building Function ---

def build_index_from_pdf(pdf_path):
    """
    Converts a PDF to markdown, chunks it, embeds it, and builds a ChromaDB vector index.
    """
    print(f"Building index from PDF: {pdf_path}")
    converter = DocumentConverter() # Using the mock converter here
    result = converter.convert(pdf_path)
    structured_text = result.document.export_to_markdown()

    # Step 1: Chunk by markdown headings
    chunks_with_ids = chunk_markdown_by_heading(structured_text)
    print(f"Initial chunks created: {len(chunks_with_ids)}")
    chunk_texts_only = [text for _, text in chunks_with_ids]
    
    # Step 2: Further chunk for embedding model token limits
    token_chunks = chunk_text_for_embedding(chunk_texts_only)

    # Create LlamaIndex Document objects
    documents = [Document(text=text) for text in token_chunks]

  
    metadatas = [{"chunk_id": str(i+1), "source_pdf": os.path.basename(pdf_path)} for i in range(len(token_chunks))]
    ids = [str(uuid4()) for _ in token_chunks]


    chroma_client = chromadb.Client()
    # Get or create a collection. Data in an in-memory client will be lost on restart.
    vector_store = ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection("my_docs"))


    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Build the VectorStoreIndex
    index = VectorStoreIndex.from_documents(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embed_model=embed_model,
        vector_store=vector_store)
    print("Index built successfully.")
    return index

# --- Reranking Function ---

def rerank_results(results, query, top_n=5):
    """
    Reranks retrieval results using a FlagEmbeddingReranker.
    """
    print(f"Reranking results for query: '{query}'")

    reranker = FlagEmbeddingReranker(
        top_n=top_n,
        model="BAAI/bge-reranker-large", # This model will be downloaded on the first run.
        use_fp16=False) # Set to True if you have a GPU and want to use half-precision
  
 
    nodes_for_reranker = [NodeWithScore(node=TextNode(text=node.get_content())) for node in results] # Extract TextNode from NodeWithScore
    query_bundle = QueryBundle(query_str=query)
    ranked_nodes = reranker._postprocess_nodes(nodes_for_reranker, query_bundle)
    print(f"Reranked {len(ranked_nodes)} nodes.")
    return ranked_nodes

# --- LLM Refinement Function ---

def llm_refine(ranked_nodes, query, groq_api_key):
    """
    Refines the answer using an LLM (Groq) based on reranked context.
    """
    print(f"Refining answer with LLM for query: '{query}'")
    reranked_texts = [node.node.get_content() for node in ranked_nodes]
    context = "\n\n".join(reranked_texts)

    # Initialize Groq LLM
    llm = Groq(api_key=groq_api_key, model="openai/gpt-oss-120b") # Using a powerful Groq model

    prompt = f"Based on the following context, answer the query comprehensively and concisely. If the context does not contain enough information, state that.\n\nQuery: {query}\n\nContext:\n{context}"
    
    # Use llm.complete for simple text generation
    response = llm.complete(prompt)
    print("LLM refinement complete.")
    print(f"LLM {response.text}")
    return response.text # Access the text attribute of the CompletionResponse


# --- Chainlit Event Handlers ---

@cl.on_chat_start
async def start():
    print(">>> START TRIGGERED")
    """
    Initializes the chat session, sets the index to None, and sends a welcome message.
    This runs every time a new chat session begins.
    """
    cl.user_session.set("index", None) # Initialize index for the session
    await cl.Message(
        content="Welcome! Please upload a PDF using the paperclip icon below. Once processed, you can start asking questions."
    ).send()

 
import asyncio
import tempfile
import os
import chainlit as cl

@cl.on_message
async def main(message: cl.Message):


    groq_api_key = "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"
    if not groq_api_key:
        await cl.Message(content="Error: GROQ_API_KEY environment variable not set. Please configure it.").send()
        return

    pdf_file_element = message.elements[0] if message.elements else None

    if pdf_file_element:
        print(f"Received PDF upload: {pdf_file_element.name}")

        # Send "processing..." message immediately
        await cl.Message(content="Processing ... This may take a moment.").send()
        await asyncio.sleep(0)
        pdf_local_path = None
        try:

           
            pdf_file_element = None
            if message.elements:
                for element in message.elements:
                    # Check if the element is a Chainlit File object and its MIME type is PDF
                    if isinstance(element, cl.File) and element.mime == "application/pdf":
                        pdf_file_element = element
                        break # Found a PDF, no need to check further

                       # --- Handle PDF Upload ---
            if pdf_file_element:
                print(type(pdf_file_element))
                print(f"Received PDF upload: {pdf_file_element.name}")

       
            # Run the blocking function in a separate thread
            index = await asyncio.to_thread(build_index_from_pdf, pdf_file_element.name)

            # Store index in the user session
            cl.user_session.set("index", index)
            
            # Notify user when done
            await cl.Message(content="PDF processed and indexed! Now you can ask your queries.").send()
        
        except Exception as e:
            await cl.Message(content=f"Error processing PDF: {e}").send()
            cl.user_session.set("index", None)

        finally:
            if pdf_local_path and os.path.exists(pdf_local_path):
                os.remove(pdf_local_path)

        return




    index = cl.user_session.get("index")
    print("ddwwwdf")
    # If no PDF has been uploaded yet (and no file was in *this* message)
    if not index:
        await cl.Message(content="No PDF has been processed yet. Please upload a PDF first using the paperclip icon.").send()
        return

    # If an index exists, proceed with the query
    query = message.content.strip() # Get the user's query and remove leading/trailing whitespace
    
    # Check if the message content is empty or just whitespace after removing attachments
    if not query:
        await cl.Message(content="Please type your question after uploading the PDF.").send()
        return

    # Perform retrieval
    retriever = index.as_retriever(similarity_top_k=10) # Retrieve top 10 relevant chunks
    results = retriever.retrieve(query)
    # print(f"Retrieved {(i for i in results)} results for query: '{query}'") 
    print(f"chunks {i}" for i in results)

    ranked_nodes = rerank_results(results, query)

    # Refine answer using LLM
    answer = llm_refine(ranked_nodes, query, groq_api_key)
    print(f"Final Answer: {answer}")





import re
import os

DEBUG_LOG_FILE = "snippet_debug.log"

def safe_display(text: str, max_len: int = 500) -> str:
    # Remove problematic control chars
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)

    # Truncate if too long
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text

async def display_snippets(ranked_nodes):
    if not ranked_nodes:
        await cl.Message(content="No relevant snippets found in the document.").send()
        return

    with open(DEBUG_LOG_FILE, "w", encoding="utf-8") as f:  # reset log each run
        f.write("=== DEBUG SNIPPETS LOG ===\n\n")

        for i, node in enumerate(ranked_nodes):
            snippet_content = node.node.get_content()
            snippet_len = len(snippet_content)

            # Log raw content to debug file
            f.write(f"\n--- Snippet {i+1} (Length: {snippet_len}, Score: {node.score:.4f}) ---\n")
            f.write(snippet_content + "\n")
            f.write("-" * 80 + "\n")

            print(f"Snippet {i+1} length: {snippet_len}")

            display_content = safe_display(snippet_content)

            try:
                await cl.Message(
                    content=f"**Snippet {i+1} (Score: {node.score:.4f}):**\n\n{display_content}"
                ).send()
            except Exception as e:
                print(f"⚠️ Failed to send snippet {i+1}: {e}")

            await asyncio.sleep(0.5)

    print(f"✅ Debug log saved at: {os.path.abspath(DEBUG_LOG_FILE)}")
