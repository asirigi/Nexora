import re
from typing import List
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

def chunk_markdown_by_heading(structured_text: str):
    lines = structured_text.strip().split('\n')
    chunks, current_chunk_lines, chunk_number = [], [], 1

    for line in lines:
        if line.strip().startswith("##") and not line.strip().startswith("###"):
            if current_chunk_lines:
                content = '\n'.join(current_chunk_lines).strip()
                chunks.append((f"chunk {chunk_number}", content))
                chunk_number += 1
                current_chunk_lines = []
        current_chunk_lines.append(line)

    if current_chunk_lines:
        content = '\n'.join(current_chunk_lines).strip()
        chunks.append((f"chunk {chunk_number}", content))
    return chunks

def _token_len(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False))

def _tail_words_for_overlap(words: List[str], overlap_tokens: int) -> List[str]:
    tail, acc = [], 0
    for w in reversed(words):
        acc += _token_len(w)
        tail.append(w)
        if acc >= overlap_tokens:
            break
    return list(reversed(tail))

def chunk_text_for_embedding(texts: List[str], max_tokens: int = 512, overlap_tokens: int = 50) -> List[str]:
    chunks = []
    for text in texts:
        if not text.strip():
            continue
        words = re.findall(r"\S+\s*", text)
        cur_words, cur_tokens, i = [], 0, 0

        while i < len(words):
            w = words[i]
            w_tok = _token_len(w)

            if w_tok > max_tokens:  # word too long, split by tokens
                if cur_words:
                    chunks.append("".join(cur_words).rstrip())
                    cur_words = _tail_words_for_overlap(cur_words, overlap_tokens)
                    cur_tokens = sum(_token_len(x) for x in cur_words)

                w_tokens = TOKENIZER.encode(w, add_special_tokens=False)
                start = 0
                while start < len(w_tokens):
                    end = min(start + max_tokens, len(w_tokens))
                    chunks.append(TOKENIZER.decode(w_tokens[start:end]))
                    if end == len(w_tokens):
                        break
                    start = end - overlap_tokens
                i += 1
                cur_words, cur_tokens = [], 0
                continue

            if cur_tokens + w_tok <= max_tokens:
                cur_words.append(w)
                cur_tokens += w_tok
                i += 1
            else:
                chunks.append("".join(cur_words).rstrip())
                cur_words = _tail_words_for_overlap(cur_words, overlap_tokens)
                cur_tokens = sum(_token_len(x) for x in cur_words)

        if cur_words:
            chunks.append("".join(cur_words).rstrip())
    return chunks
