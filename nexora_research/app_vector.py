import chainlit as cl
from docling.document_converter import DocumentConverter
from llama_index.core.schema import Document, NodeWithScore, QueryBundle, TextNode
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import chromadb
from uuid import uuid4
import nltk
from transformers import AutoTokenizer

# Uncomment if running for the first time
# nltk.download('punkt')

def chunk_markdown_by_heading(structured_text):
    lines = structured_text.strip().split('\n')
    chunks = []
    current_chunk = []
    chunk_number = 1
    for line in lines:
        if line.strip().startswith("##") and not line.strip().startswith("###"):
            if current_chunk:
                content = '\n'.join(current_chunk).strip()
                chunks.append((f"chunk {chunk_number}", content))
                chunk_number += 1
                current_chunk = []
        current_chunk.append(line)
    if current_chunk:
        content = '\n'.join(current_chunk).strip()
        chunks.append((f"chunk {chunk_number}", content))
    return chunks

def chunk_text_for_embedding(chunk_texts, model_name="BAAI/bge-base-en-v1.5", max_tokens=512, buffer_tokens=20):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_chunks = []
    for i, text in enumerate(chunk_texts):
        sentences = nltk.sent_tokenize(text)
        current_chunk = []
        current_tokens = 0
        for sentence in sentences:
            token_count = len(tokenizer.encode(sentence, add_special_tokens=False))
            if current_tokens + token_count <= max_tokens - buffer_tokens:
                current_chunk.append(sentence)
                current_tokens += token_count
            else:
                all_chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = token_count
        if current_chunk:
            all_chunks.append(" ".join(current_chunk))
    return all_chunks

def build_index_from_pdf(pdf_path):
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    structured_text = result.document.export_to_markdown()
    chunks = chunk_markdown_by_heading(structured_text)
    chunk_texts = [text for _, text in chunks]
    token_chunks = chunk_text_for_embedding(chunk_texts)
    documents = [Document(text=text) for text in token_chunks]
    metadatas = [{"chunk_id": str(i+1), "source": pdf_path} for i in range(len(token_chunks))]
    ids = [str(uuid4()) for _ in token_chunks]
    chroma_client = chromadb.Client()
    vector_store = ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection("my_docs"))
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_documents(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embed_model=embed_model,
        vector_store=vector_store
    )
    return index

def rerank_results(results, query, top_n=5):
    reranker = FlagEmbeddingReranker(
        top_n=top_n,
        model="BAAI/bge-reranker-large",
        use_fp16=False
    )
    nodes = [NodeWithScore(node=TextNode(text=node.get_content())) for node in results]
    query_bundle = QueryBundle(query_str=query)
    ranked_nodes = reranker._postprocess_nodes(nodes, query_bundle)
    return ranked_nodes

def llm_refine(ranked_nodes, query, groq_api_key):
    reranked_texts = [node.node.get_content() for node in ranked_nodes]
    context = "\n\n".join(reranked_texts)
    llm = Groq(api_key=groq_api_key, model="llama3-70b-8192")
    prompt = f"Based on the following context, answer the query: {query}\n\n{context}"
    response = llm.complete(prompt)
    return response

# Chainlit UI
@cl.on_chat_start
async def start():
    cl.user_session.set("index", None)
    pdf_path = cl.user_session.get("pdf_path")
    await cl.Message(content="Welcome! Upload a PDF using the paperclip icon below, then ask your question.").send()

    files = None

    # Wait for user to upload csv data
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a csv file to begin!", 
            accept=["pdf"],
            max_size_mb= 100,
            timeout = 180,
        ).send()

    # load the csv data and store in user_session
    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    index = cl.user_session.get("index")
    index = build_index_from_pdf(pdf_path = file.path)


@cl.on_message
async def main(message: cl.Message):
    groq_api_key = "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"
    index = cl.user_session.get("index")

    # Handle file upload (PDF)
    if hasattr(message, "attachments") and message.attachments:
        pdf_file = message.attachments[0]
        pdf_path = pdf_file.path
        index = build_index_from_pdf(pdf_path)
        cl.user_session.set("index", index)
        await cl.Message(content="PDF processed and indexed! Now ask your query.").send()
        return

    # If no PDF has been uploaded yet
    if not index:
        await cl.Message(content="Please upload a PDF first using the paperclip icon.").send()
        return

    # Handle query
    query = message.content
    retriever = index.as_retriever(similarity_top_k=10)
    results = retriever.retrieve(query)
    ranked_nodes = rerank_results(results, query)
    answer = llm_refine(ranked_nodes, query, groq_api_key)

    for node in ranked_nodes:
        await cl.Message(content=f"Score: {node.score:.4f}\n\n{node.node.get_content()}").send()

    await cl.Message(content=f"LLM Refined Answer:\n\n{answer}").send()