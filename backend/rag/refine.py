from llama_index.llms.groq import Groq
import os
# import os
# from llama_index.llms.azure_openai import AzureOpenAI
from pydantic import NonNegativeInt
from dotenv import load_dotenv
load_dotenv() 


def llm_refine(ranked_nodes, query, groq_api_key):
    reranked_texts = [node.node.get_content() for node in ranked_nodes]
    context = "\n\n".join(reranked_texts)
    llm = Groq(api_key=groq_api_key, model="openai/gpt-oss-120b")

    prompt = f"""Based on the following context, answer the query comprehensively and concisely. 
    If the context does not contain enough information, state that.\n\nQuery: {query}\n\nContext:\n{context}"""

    response = llm.complete(prompt)
    return response.text


def chat_with_llm(query, groq_api_key):
    # reranked_texts = [node.node.get_content() for node in ranked_nodes]
    # context = "\n\n".join(reranked_texts)
    llm = Groq(api_key=groq_api_key, model="openai/gpt-oss-120b")

    prompt = f"""Based on the following context, answer the query comprehensively and concisely, like a normal conversation.{query}\n\n"""

    response = llm.complete(prompt)
    return response.text

# ----------------------------------------------------------------------------------------------
# Refine and Chat functions with chat history support

# def llm_refine(ranked_nodes, query, groq_api_key, chat_history=None):
#     reranked_texts = [node.node.get_content() for node in ranked_nodes]
#     context = "\n\n".join(reranked_texts)
#     llm = Groq(api_key=groq_api_key, model="openai/gpt-oss-120b")

#     history_text = ""
#     if chat_history:
#         history_text = "\n".join([f"{t['role']}: {t['content']}" for t in chat_history])

#     prompt = f"""You are a helpful assistant. Use the conversation history and provided context
#     to answer the user query.

#     Conversation history:
#     {history_text}

#     Context:
#     {context}

#     Query: {query}
#     """

#     response = llm.complete(prompt)
#     return response.text


# def chat_with_llm(query, groq_api_key, chat_history=None):
#     llm = Groq(api_key=groq_api_key, model="openai/gpt-oss-120b")

#     history_text = ""
#     if chat_history:
#         history_text = "\n".join([f"{t['role']}: {t['content']}" for t in chat_history])

#     prompt = f"""You are a helpful assistant. Continue the conversation naturally.

#     Conversation history:
#     {history_text}

#     User: {query}
#     Assistant:
#     """

#     response = llm.complete(prompt)
#     return response.text


# -----------------------------------------------------------------------------------------------
# Refine and Chat functions with chat history support for Azure OpenAI



# def llm_refine(ranked_nodes, query, chat_history=None):
#     reranked_texts = [node.node.get_content() for node in ranked_nodes]
#     context = "\n\n".join(reranked_texts)
    
    
#     # # Load Azure credentials from environment variables
#     # azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     # azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
#     # azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
#     # azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")



# # # Load Azure credentials from environment variables
# # azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://my-azure-openai.openai.azure.com/")
# # azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "abcd1234...")
# # azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt4-mini-deployment")
# # azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
#     deployment_name = "gpt-4.1-mini"  


# # Initialize the AzureOpenAI LLM
#     llm = AzureOpenAI(
#         model="gpt-4.1-mini",    # family name
#         engine=deployment_name,  # your deployment name
#         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#         api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     )

#     history_text = ""
#     # if chat_history:
#     #     history_text = "\n".join([f"{t['role']}: {t['content']}" for t in chat_history])

#     prompt = f"""You are a helpful assistant. Use the conversation history and provided context
#     to answer the user query.

#     Conversation history:
#     {history_text}

#     Context:
#     {context}

#     Query: {query}
#     """
    
#     # Example: Ask a question
#     # prompt = "Explain quantum computing in very simple terms."
#     # response = llm.complete(prompt)
        
#     # print("Response:", response.text)


#     response = llm.complete(prompt)
#     return response.text


# # def test_azure_llm():
    
# #     # deployment_name = "gpt-4-1-mini"  
# #     # api_key = ""
# #     # endpoint ="https://praval-ds.openai.azure.com/"
# #     # api_version = "2024-12-01-preview"
# #     # model="gpt-4.1-mini",      # model family
# #     # engine=deployment_name,    # your Azure deployment name
# #     # api_key=
# #     # azure_endpoint=,
# #     # api_version=,

# #     deployment_name = "gpt-4.1-mini"  # ⚠️ must match Azure Portal deployment

# #     llm = AzureOpenAI(
# #         model="gpt-4.1-mini",    # family name
# #         engine=deployment_name,  # your deployment name
# #         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
# #         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
# #         api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
# #     )

# #     response = llm.complete("Hello, can you confirm you're working?")
# #     print("✅ Azure LLM Response:", response.text)



# def chat_with_llm(query, groq_api_key, chat_history=None):
#     llm = Groq(api_key=groq_api_key, model="openai/gpt-oss-120b")

#     history_text = ""
#     # if chat_history is None:
#     #     history_text = "\n".join([f"{t['role']}: {t['content']}" for t in chat_history])
#     # else:
#     #     history_text = ""
#     prompt = f"""You are a helpful assistant. Continue the conversation naturally.

#     Conversation history:
#     {history_text}

#     User: {query}
#     Assistant:
#     """

#     response = llm.complete(prompt)
#     return response.text























