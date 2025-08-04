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

# --- Mock for docling.document_converter ---
# IMPORTANT: You need to replace this mock with your actual docling.document_converter
# or a suitable PDF processing library (e.g., LlamaIndex's PdfReader, PyPDF2, pypdf).
# This mock allows the code to run without docling installed.
# class MockDocument:
#     def export_to_markdown(self):
#         # Return some mock markdown content for testing
#         return """
# # Document Title

# ## Section 1: Introduction
# This is the introduction to the document. It talks about various concepts.
# This section is quite informative and provides a good overview.

# ## Section 2: Key Concepts
# Here we delve into the key concepts.
# Concept A is very important.
# Concept B builds upon Concept A.

# ## Section 3: Conclusion
# In conclusion, this document covers essential topics.
# Further reading is recommended.
# """

# class MockDocumentConverter:
#     def convert(self, pdf_path):
#         print(f"MockDocumentConverter: Converting {pdf_path}")
#         # In a real scenario, this would parse the PDF and extract content
#         class MockResult:
#             def __init__(self):
#                 self.document = MockDocument()
#         return MockResult()

# Replace the actual import with the mock for runnability if docling is not installed
# from docling.document_converter import DocumentConverter
# DocumentConverter = MockDocumentConverter


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
            print(all_final_chunks)
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

    # Generate unique IDs and metadata for each document
    # Note: ChromaDB collection name is fixed for this example ("my_docs").
    # For persistent storage, you might want to manage collection names dynamically.
    metadatas = [{"chunk_id": str(i+1), "source_pdf": os.path.basename(pdf_path)} for i in range(len(token_chunks))]
    ids = [str(uuid4()) for _ in token_chunks]

    # Initialize ChromaDB client (in-memory for this example)
    # For persistent storage, you would configure a persistent client:
    # chromadb.PersistentClient(path="/path/to/your/db")
    chroma_client = chromadb.Client()
    # Get or create a collection. Data in an in-memory client will be lost on restart.
    vector_store = ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection("my_docs"))

    # Initialize embedding model
    # This model will be downloaded on the first run.
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
  
    # Convert LlamaIndex NodeWithScore objects to TextNode for reranker
    # The reranker expects a list of nodes, not directly NodeWithScore from retriever
    # It will re-wrap them with scores after reranking
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
    llm = Groq(api_key=groq_api_key, model="llama3-70b-8192") # Using a powerful Groq model

    prompt = f"Based on the following context, answer the query comprehensively and concisely. If the context does not contain enough information, state that.\n\nQuery: {query}\n\nContext:\n{context}"
    
    # Use llm.complete for simple text generation
    response = llm.complete(prompt)
    print("LLM refinement complete.")
    return response.text # Access the text attribute of the CompletionResponse


# --- Chainlit Event Handlers ---

@cl.on_chat_start
async def start():
    """
    Initializes the chat session, sets the index to None, and sends a welcome message.
    This runs every time a new chat session begins.
    """
    cl.user_session.set("index", None) # Initialize index for the session
    await cl.Message(
        content="Welcome! Please upload a PDF using the paperclip icon below. Once processed, you can start asking questions."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """
    Handles incoming messages.
    Detects PDF uploads to build an index, or processes user queries if an index exists.
    """
    # Get Groq API key from environment variable.
    # It's highly recommended to set this as an environment variable in production.
    groq_api_key = ""
    if not groq_api_key:
        await cl.Message(content="Error: GROQ_API_KEY environment variable not set. Please configure it.").send()
        return

    # Check if there are any file attachments in the current message
    pdf_file_element = None
    if message.elements:
        for element in message.elements:
            # Check if the element is a Chainlit File object and its MIME type is PDF
            if isinstance(element, cl.File) and element.mime == "application/pdf":
                pdf_file_element = element
                break # Found a PDF, no need to check further

    # --- Handle PDF Upload ---
    if pdf_file_element:
        # Check if the content is actually available
        # if pdf_file_element.content is None:
        #     await cl.Message(content=f"Error: Could not read content of `{pdf_file_element.name}`. Please try uploading the file again.").send()
        #     print(f"Error: pdf_file_element.content is None for {pdf_file_element.name}")
        #     return # Exit if content is not available

        await cl.Message(content=f"Processing `{pdf_file_element.name}`... This may take a moment.").send()
        dj = pdf_file_element.name
        # pdf_local_path = None # Initialize to None
        # try:
        #     # Create a temporary file to save the uploaded PDF content
        #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        #         tmp_file.write(pdf_file_element.content)
        #         pdf_local_path = tmp_file.name # Get the path to the temporary file

        #     # Build the LlamaIndex from the temporary PDF file
        index = build_index_from_pdf(dj) 
        cl.user_session.set("index", index) # Store the index in the user's session

        await cl.Message(content="PDF processed and indexed! Now you can ask your queries.").send()

        # except Exception as e:
        #     # Handle any errors during PDF processing
        #     error_message = f"Error processing PDF: {e}\n\nPlease try again or upload a different file."
        #     await cl.Message(content=error_message).send()
        #     print(f"Detailed error during PDF processing: {e}")
        #     # Ensure the index is reset if processing fails
        #     cl.user_session.set("index", None)
        # finally:
        #     # Clean up the temporary file regardless of success or failure
        #     if pdf_local_path and os.path.exists(pdf_local_path):
        #         os.remove(pdf_local_path)
        
        # IMPORTANT: Exit the function after handling the file upload.
        # This prevents the code from trying to process `message.content` as a query
        # when the user has just uploaded a file.
        return

    # --- Handle Query (if no file was uploaded in this message) ---
    # Retrieve the index from the user's session
    index = cl.user_session.get("index")

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
    # Perform reranking
    # Pass the original query to reranker for better context
    ranked_nodes = rerank_results(results, query)

    # Refine answer using LLM
    answer = llm_refine(ranked_nodes, query, groq_api_key)

    # Display retrieved snippets (optional, for debugging/transparency)
    if ranked_nodes:
        await cl.Message(content="**Relevant Snippets from PDF:**").send()
        for i, node in enumerate(ranked_nodes):
            # Truncate content for cleaner display in chat
            snippet_content = node.node.get_content()
            display_content = snippet_content[:500] + "..." if len(snippet_content) > 500 else snippet_content
            await cl.Message(content=f"**Snippet {i+1} (Score: {node.score:.4f}):**\n\n{display_content}").send()
    else:
        await cl.Message(content="No relevant snippets found in the document.").send()

    # Display the final refined answer
    await cl.Message(content=f"**LLM Refined Answer:**\n\n{answer}").send()
