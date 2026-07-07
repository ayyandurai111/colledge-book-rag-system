"""
Tokenizer — accurate BPE-approximate token counting, no network required.

Uses tiktoken when pre-cached. Falls back to a calibrated local estimator.
"""
import re
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MAX_TOKENS = 512
_enc        = None
_TIKTOKEN_OK = False

try:
    import tiktoken as _tiktoken
    _enc = _tiktoken.get_encoding("cl100k_base")
    _TIKTOKEN_OK = True
    logger.info("Tokenizer: tiktoken cl100k_base loaded")
except Exception:
    logger.info("Tokenizer: using offline word-split approximation")

# Matches words (with contractions), numbers, and punctuation
_TOK_RE = re.compile(r"[A-Za-z]+(?:'[a-z]+)*|[0-9]+|[^A-Za-z0-9\s]")


def _local_count(text: str) -> int:
    """
    Approximate GPT-4 BPE count using word-split heuristic.
    Calibration:
      - "hello world"         → 2   (matches tiktoken)
      - typical English word  → 1 token if ≤6 chars, else 2
      - numbers/punctuation   → 1 token each
    """
    if not text:
        return 0
    tokens = _TOK_RE.findall(text)
    count  = sum(1 + (len(t) > 6) for t in tokens)
    return max(1, count)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_enc.encode(text)) if _TIKTOKEN_OK else _local_count(text)


def truncate_to_limit(text: str, max_tokens: int = _MAX_TOKENS) -> str:
    """
    Hard-truncate text to max_tokens BPE tokens.
    Tries to end on a sentence boundary (., !, ?).
    """
    if count_tokens(text) <= max_tokens:
        return text

    if _TIKTOKEN_OK and _enc is not None:
        truncated = _enc.decode(_enc.encode(text)[:max_tokens])
    else:
        # Local fallback: binary-search on character length
        # Average English: ~4.5 chars/token; start conservative
        lo, hi = 0, len(text)
        guess  = min(len(text), max_tokens * 4)
        for _ in range(20):        # converge in ≤20 iterations
            mid = (lo + hi) // 2
            if count_tokens(text[:mid]) <= max_tokens:
                lo = mid
            else:
                hi = mid
        truncated = text[:lo]

    # Prefer sentence boundary
    for sep in (".", "!", "?"):
        last = truncated.rfind(sep)
        if last > len(truncated) // 2:
            return truncated[: last + 1].strip()

    return truncated.strip()
