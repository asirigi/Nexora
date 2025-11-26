

# import re
# import os

# DEBUG_LOG_FILE = "snippet_debug.log"

# def safe_display(text: str, max_len: int = 500) -> str:
#     # Remove problematic control chars
#     text = re.sub(r"[\x00-\x1F\x7F]", "", text)

#     # Truncate if too long
#     if len(text) > max_len:
#         return text[:max_len] + "..."
#     return text

# async def display_snippets(ranked_nodes):
#     if not ranked_nodes:
#         await cl.Message(content="No relevant snippets found in the document.").send()
#         return

#     with open(DEBUG_LOG_FILE, "w", encoding="utf-8") as f:  # reset log each run
#         f.write("=== DEBUG SNIPPETS LOG ===\n\n")

#         for i, node in enumerate(ranked_nodes):
#             snippet_content = node.node.get_content()
#             snippet_len = len(snippet_content)

#             # Log raw content to debug file
#             f.write(f"\n--- Snippet {i+1} (Length: {snippet_len}, Score: {node.score:.4f}) ---\n")
#             f.write(snippet_content + "\n")
#             f.write("-" * 80 + "\n")

#             print(f"Snippet {i+1} length: {snippet_len}")

#             display_content = safe_display(snippet_content)

#             try:
#                 await cl.Message(
#                     content=f"**Snippet {i+1} (Score: {node.score:.4f}):**\n\n{display_content}"
#                 ).send()
#             except Exception as e:
#                 print(f"⚠️ Failed to send snippet {i+1}: {e}")

#             await asyncio.sleep(0.5)

#     print(f"✅ Debug log saved at: {os.path.abspath(DEBUG_LOG_FILE)}")





# -----------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------




    # # Display retrieved snippets (optional, for debugging/transparency)
    # if ranked_nodes:
    #     print(type(ranked_nodes))
    #     # print(f" nodes 1 ---------------------------------  {ranked_nodes}")

    #     # print(f"Displaying top {len(ranked_nodes)} relevant snippets:")
    #     await cl.Message(content="**Relevant Snippets from PDF:**").send()
    #     await asyncio.sleep(1)


    #     for i, node in enumerate(ranked_nodes):
    #         print("successs")

    #         # Truncate content for cleaner display in chat
    #         snippet_content = node.node.get_content()
    #         print("#"*80)

    #         print(f" content --- {snippet_content}")
    #         print(f"#*{40} content over")
    #         if len(snippet_content) > 500:
    #                  print("dfff")
    #                  display_content = snippet_content[:500] + "..."
    #         else:
    #            display_content = snippet_content
    #         await cl.Message(content=f"**Snippet {i+1} {display_content}").send()
    #         await asyncio.sleep(1)
    #         print("dfjs")
    # else:
    #     await cl.Message(content="No relevant snippets found in the document.").send()
