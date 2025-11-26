from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

def rerank_results(results, query, top_n=5):
    reranker = FlagEmbeddingReranker(
        top_n=top_n,
        model="BAAI/bge-reranker-large",
        use_fp16=False
    )
    nodes_for_reranker = [NodeWithScore(node=TextNode(text=node.get_content())) for node in results]
    query_bundle = QueryBundle(query_str=query)
    return reranker._postprocess_nodes(nodes_for_reranker, query_bundle)
