"""
Embedder — NVIDIA NV-Embed-v1
Fix 3: truncate chunks to 512-token hard limit before sending to API.
"""
from openai import OpenAI
from src.utils.config import get_config, get_nvidia_api_key
from src.utils.logger import get_logger
from src.utils.tokenizer import truncate_to_limit, count_tokens

logger = get_logger(__name__)

_client = None
MAX_EMBED_TOKENS = 512   # NV-Embed-v1 hard limit


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=get_nvidia_api_key(),
            base_url="https://integrate.api.nvidia.com/v1"
        )
    return _client


def embed_texts(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """
    Embed texts with NVIDIA NV-Embed-v1.
    Each text is truncated to 512 tokens before sending (Fix 3).
    """
    config     = get_config()
    model      = config["embedding"]["model"]
    batch_size = config["embedding"].get("batch_size", 50)
    client     = get_client()

    # Truncate every text to the model's hard limit
    safe_texts = []
    truncated_count = 0
    for t in texts:
        if count_tokens(t) > MAX_EMBED_TOKENS:
            safe_texts.append(truncate_to_limit(t, MAX_EMBED_TOKENS))
            truncated_count += 1
        else:
            safe_texts.append(t)

    if truncated_count:
        logger.warning(f"Truncated {truncated_count}/{len(texts)} texts to {MAX_EMBED_TOKENS} tokens")

    all_embeddings = []
    total_batches  = (len(safe_texts) + batch_size - 1) // batch_size

    for i in range(0, len(safe_texts), batch_size):
        batch     = safe_texts[i: i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")

        response = client.embeddings.create(
            input=batch,
            model=model,
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "END"}  # safety net
        )
        all_embeddings.extend(item.embedding for item in response.data)

    logger.info(f"Generated {len(all_embeddings)} embeddings")
    return all_embeddings


def embed_chunks(chunks: list[dict]) -> list[dict]:
    texts      = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, input_type="passage")
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def embed_query(query: str) -> list[float]:
    safe_query = truncate_to_limit(query, MAX_EMBED_TOKENS)
    logger.info(f"Embedding query ({count_tokens(safe_query)} tokens): {safe_query[:80]}...")
    return embed_texts([safe_query], input_type="query")[0]
