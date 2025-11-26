import os
from uuid import uuid4
from docling.document_converter import DocumentConverter
from llama_index.core.schema import Document

from .chunking import chunk_markdown_by_heading, chunk_text_for_embedding
from .embedding import get_embedding_model
from .retriever import create_index

def build_index_from_pdf(pdf_path: str):
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    structured_text = result.document.export_to_markdown()

    chunks_with_ids = chunk_markdown_by_heading(structured_text)
    chunk_texts_only = [text for _, text in chunks_with_ids]
    token_chunks = chunk_text_for_embedding(chunk_texts_only)

    documents = [Document(text=text) for text in token_chunks]
    metadatas = [{"chunk_id": str(i+1), "source_pdf": os.path.basename(pdf_path)} for i in range(len(token_chunks))]
    ids = [str(uuid4()) for _ in token_chunks]

    embed_model = get_embedding_model()
    index = create_index(documents, metadatas, ids, embed_model)
    return index
