import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex

def create_index(documents, metadatas, ids, embed_model):
    chroma_client = chromadb.Client()
    vector_store = ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection("my_docs"))
    index = VectorStoreIndex.from_documents(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embed_model=embed_model,
        vector_store=vector_store
    )
    return index
