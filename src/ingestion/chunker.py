"""
Textbook-Aware Chunker
======================
Strategies:
  1. TOC-based   → exact section boundaries from PDF bookmarks
  2. Font-based  → heading hierarchy from font-size + bold + ALL-CAPS heuristics

Chunk schema:
{
  chunk_id, source, chapter, section, subsection,
  page_number, heading_path, chunk_type, text, token_count
}
"""

import re
import hashlib
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.tokenizer import count_tokens, truncate_to_limit

logger = get_logger(__name__)

# ─── Special-content detectors ───────────────────────────────────────────────
_DEFINITION_RE = re.compile(
    r"^\s*(definition|def\.?|theorem|lemma|corollary|axiom|proposition)\b",
    re.IGNORECASE
)
_FORMULA_RE = re.compile(r"[\$\\∑∏∫√±×÷≠≤≥∞∂]|\\[a-zA-Z]+\{")
_TABLE_RE   = re.compile(r"^(table\s+\d+[\.\:])", re.IGNORECASE)
_FIGURE_RE  = re.compile(r"^(figure\s+\d+[\.\:]|fig\.?\s*\d+[\.\:])", re.IGNORECASE)

# Matches a heading whose label IS the content-type signal (e.g. a subsection
# literally titled "Definition", "Theorem 2.1", "Formula", "Table", "Figure")
# even when the prose body text itself doesn't start with that word.
_HEADING_TYPE_RE = re.compile(
    r"^\s*(definition|theorem|lemma|corollary|axiom|proposition)\b",
    re.IGNORECASE
)
_HEADING_FORMULA_RE = re.compile(r"^\s*(formula|equation)s?\b", re.IGNORECASE)
_HEADING_TABLE_RE   = re.compile(r"^\s*table\b", re.IGNORECASE)
_HEADING_FIGURE_RE  = re.compile(r"^\s*(figure|fig\.?)\b", re.IGNORECASE)


def _detect_chunk_type(text: str, heading_label: str = "") -> str:
    """
    Classify chunk content type.

    Checks the prose body first (explicit "Definition:", LaTeX-like formula
    markers, "Table N.", "Figure N." prefixes). Falls back to the nearest
    heading label (subsection/section name) when the body text itself
    doesn't carry the signal — common in textbooks where the author labels
    a subsection "Definition" and the quoted prose underneath doesn't
    repeat that word (e.g. "According to Peter Drucker, ...").
    """
    s = text.strip()
    if _DEFINITION_RE.match(s): return "definition"
    if _FORMULA_RE.search(s):   return "formula"
    if _TABLE_RE.match(s):      return "table"
    if _FIGURE_RE.match(s):     return "figure"

    if heading_label:
        h = heading_label.strip()
        if _HEADING_TYPE_RE.match(h):    return "definition"
        if _HEADING_FORMULA_RE.match(h): return "formula"
        if _HEADING_TABLE_RE.match(h):   return "table"
        if _HEADING_FIGURE_RE.match(h):  return "figure"

    return "text"


# ─── Heading heuristics (font-based fallback) ─────────────────────────────────
# ─── Strong structural heading patterns (font-size-independent) ──────────────
_UNIT_CHAPTER_RE = re.compile(
    r"^\s*(UNIT|CHAPTER|MODULE)\s*[\-–:]?\s*([IVXLCDM]+|\d+)\b", re.IGNORECASE
)


def _is_heading_line(text: str, font_size: float, is_bold: bool,
                     body_size: float) -> str | None:
    """
    Return heading level ('chapter','section','subsection') or None.

    Three signals, checked in priority order:
      1. Structural pattern ("UNIT I", "CHAPTER 3") → always 'chapter',
         regardless of font size — handles textbooks where unit titles
         are not visually distinguished from body text (common in
         Word-exported study materials).
      2. Font-size thresholds relative to body text.
      3. Bold/ALL-CAPS short lines → 'subsection', but only if the line
         doesn't look like a quoted sentence (no mid-line period before
         a lowercase word, no trailing quotation mark) to avoid treating
         bolded quote-introductions ("According to Peter Drucker, ...")
         as headings.
    """
    stripped = text.strip()
    if not stripped:
        return None

    if _UNIT_CHAPTER_RE.match(stripped):
        return "chapter"

    short_line = len(stripped) < 80
    all_caps   = stripped.isupper() and len(stripped) > 3

    if font_size >= body_size + 4:
        return "chapter"
    if font_size >= body_size + 2:
        return "section"

    # Guard against bolded sentence fragments / quote leads being misread
    # as subsection headings: real headings don't end with a quote mark
    # or contain ", " followed by lowercase (a clause continuation).
    looks_like_prose = (
        stripped.endswith(('"', "'", ".", ",")) or
        bool(re.search(r",\s+[a-z]", stripped))
    )

    if (is_bold or all_caps) and short_line and not looks_like_prose \
            and font_size >= body_size - 0.5:
        return "subsection"
    return None


# ─── Sentence splitting ───────────────────────────────────────────────────────
def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


# ─── Token-aware sliding window ───────────────────────────────────────────────
def _window_sentences(sentences: list[str], min_tok: int, max_tok: int,
                      overlap_tok: int) -> list[str]:
    chunks, current, current_tok = [], [], 0

    def flush():
        if current:
            chunks.append(" ".join(current))

    for sent in sentences:
        tok = count_tokens(sent)
        if current_tok + tok > max_tok and current:
            flush()
            overlap, ov_tok = [], 0
            for s in reversed(current):
                t = count_tokens(s)
                if ov_tok + t > overlap_tok:
                    break
                overlap.insert(0, s)
                ov_tok += t
            current    = overlap
            current_tok = ov_tok
        current.append(sent)
        current_tok += tok

    flush()
    return chunks


# ─── Stable chunk ID via content hash (Fix 4: deduplication) ─────────────────
def _make_chunk_id(source: str, text: str, page: int = 0, index: int = 0) -> str:
    """
    Deterministic, collision-resistant ID.

    Combines content hash + (page, index-within-page) so that:
      - Re-ingesting the SAME pdf produces the SAME ids → upsert dedups correctly.
      - Two chunks with identical text but different positions get different ids
        (e.g. repeated boilerplate/headers across pages).
    """
    key    = f"{source}:{page}:{index}:{text[:200]}:{len(text)}".encode("utf-8")
    digest = hashlib.md5(key, usedforsecurity=False).hexdigest()[:12]
    return f"{source}_{digest}"


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_document(doc: dict) -> list[dict]:
    if doc["pdf_type"] == "toc":
        chunks = _chunk_by_toc(doc)
    else:
        chunks = _chunk_by_font(doc)

    validated = []
    for chunk in chunks:
        chunk = _add_heading_prefix(chunk)
        result = _quality_check(chunk)
        if result:
            validated.append(result)

    logger.info(
        f"[{doc['source']}] {len(validated)} valid chunks "
        f"({len(chunks) - len(validated)} dropped)"
    )
    return validated


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: TOC-BASED
# ═══════════════════════════════════════════════════════════════════════════════

def _chunk_by_toc(doc: dict) -> list[dict]:
    config  = get_config()
    min_tok = config["chunking"]["min_tokens"]
    max_tok = config["chunking"]["max_tokens"]
    overlap = config["chunking"]["overlap_tokens"]
    source  = doc["source"]
    toc     = doc["toc"]
    pages   = doc["pages"]

    page_text   = {p["page"]: p["text"] for p in pages}
    total_pages = max(page_text.keys()) if page_text else 1

    sections = []
    for i, entry in enumerate(toc):
        level, title, start = entry[0], entry[1], entry[2]
        end = toc[i + 1][2] - 1 if i + 1 < len(toc) else total_pages
        sections.append({"level": level, "title": title, "start": start, "end": end})

    def _resolve_path(idx):
        chapter = section = subsection = ""
        for j in range(idx + 1):
            lvl, ttl = sections[j]["level"], sections[j]["title"]
            if lvl == 1:   chapter, section, subsection = ttl, "", ""
            elif lvl == 2: section, subsection = ttl, ""
            elif lvl >= 3: subsection = ttl
        return chapter, section, subsection

    all_chunks = []
    global_idx = 0   # running counter across whole doc — deterministic per identical input
    for idx, sec in enumerate(sections):
        chapter, section, subsection = _resolve_path(idx)
        section_text = "\n".join(
            page_text.get(pg, "")
            for pg in range(sec["start"], sec["end"] + 1)
        ).strip()
        if not section_text:
            continue

        windows = _window_sentences(_split_sentences(section_text), min_tok, max_tok, overlap)
        for win in windows:
            all_chunks.append({
                "chunk_id":    _make_chunk_id(source, win, sec["start"], global_idx),
                "source":      source,
                "chapter":     chapter,
                "section":     section,
                "subsection":  subsection,
                "page_number": sec["start"],
                "heading_path": _build_heading_path(chapter, section, subsection),
                "chunk_type":  _detect_chunk_type(win, subsection or section),
                "text":        win,
                "token_count": count_tokens(win),
                # Deterministic position of this window within the document —
                # lets retrieval-time context expansion walk forward/backward
                # from a hit and reconstruct the full section in order.
                "chunk_index": global_idx
            })
            global_idx += 1
    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: FONT-BASED
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_running_headers(pages: list[dict], min_repeat_ratio: float = 0.5,
                            max_header_chars: int = 100,
                            position_tolerance: float = 5.0) -> set[str]:
    """
    Find lines that repeat near-identically across many pages AND sit in
    a near-identical page position — these are running headers/footers
    (e.g. "STUDY MATERIAL FOR BBA", page banners) rather than real
    content or headings, and should be excluded from both the body text
    and heading detection.

    Position consistency (via bbox y-coordinate) is the primary signal:
    real banners render at the same spot on every page, whereas a
    legitimate recurring structural heading (e.g. a "Definition"
    subsection appearing in every unit) shifts position based on how
    much content precedes it. Text-repetition alone would wrongly treat
    such headings as banners; requiring consistent position avoids that.

    Falls back to text-repetition-only (original heuristic) when bbox
    data isn't available, to remain robust to different extraction paths.
    """
    from collections import Counter, defaultdict
    line_counts = Counter()
    line_y_positions: dict[str, list[float]] = defaultdict(list)
    total_pages = len(pages)
    if total_pages == 0:
        return set()

    for page in pages:
        seen_this_page = set()
        for block in page["blocks"]:
            norm = re.sub(r"\s+", " ", block["text"].strip()).upper()
            if norm and len(norm) <= max_header_chars and norm not in seen_this_page:
                line_counts[norm] += 1
                seen_this_page.add(norm)
                bbox = block.get("bbox")
                if bbox and len(bbox) >= 2:
                    line_y_positions[norm].append(bbox[1])  # y0 coordinate

    threshold = max(3, int(total_pages * min_repeat_ratio))
    candidates = {line for line, count in line_counts.items() if count >= threshold}

    headers = set()
    for line in candidates:
        positions = line_y_positions.get(line, [])
        if len(positions) >= 2:
            # Cluster positions and require most occurrences to agree —
            # robust to a few outlier pages (e.g. a differently-styled
            # cover page) that would otherwise break a simple min/max
            # range check on an avg otherwise-consistent banner.
            rounded = [round(p / position_tolerance) * position_tolerance for p in positions]
            mode_count = Counter(rounded).most_common(1)[0][1]
            if mode_count / len(positions) >= 0.7:
                headers.add(line)
            # else: position too inconsistent → likely a recurring
            # structural heading, not a banner — don't flag it
        else:
            # No bbox data available — fall back to text-repetition only
            headers.add(line)

    return headers


def _is_running_header(text: str, running_headers: set[str]) -> bool:
    norm = re.sub(r"\s+", " ", text.strip()).upper()
    if norm in running_headers:
        return True
    # Bare page numbers
    if re.fullmatch(r"\d{1,4}", norm):
        return True
    return False


def _is_toc_listing_line(text: str) -> bool:
    """Detect the literal 'Table of Content' label and its listing rows."""
    return bool(re.match(r"^\s*table\s+of\s+content", text.strip(), re.IGNORECASE))


def _chunk_by_font(doc: dict) -> list[dict]:
    config  = get_config()
    min_tok = config["chunking"]["min_tokens"]
    max_tok = config["chunking"]["max_tokens"]
    overlap = config["chunking"]["overlap_tokens"]
    source  = doc["source"]
    pages   = doc["pages"]

    all_sizes = [b["font_size"] for p in pages for b in p["blocks"] if b["font_size"] > 0]
    if not all_sizes:
        logger.warning("No font-size data; plain chunking")
        return _plain_chunk_fallback(doc)

    all_sizes.sort()
    body_size = all_sizes[len(all_sizes) // 2]
    logger.info(f"[{source}] Body font size: {body_size:.1f}pt")

    running_headers = _detect_running_headers(pages)
    if running_headers:
        logger.info(f"[{source}] Detected {len(running_headers)} running header/footer lines to skip")

    cur_chapter = cur_section = cur_subsection = ""
    cur_page      = 1
    buffer_page   = 1     # page where the buffer's content STARTED (fixes mis-stamping)
    buffer: list[str] = []
    all_chunks: list[dict] = []
    global_idx = 0        # running counter across the whole document — deterministic per doc
    in_toc_listing = False   # once we hit "Table of Content", suppress heading capture
                              # until the next UNIT/CHAPTER pattern resets it

    def flush_buffer():
        nonlocal global_idx
        if not buffer:
            return
        text      = " ".join(buffer)
        sentences = _split_sentences(text)
        windows   = _window_sentences(sentences, min_tok, max_tok, overlap)
        for win in windows:
            all_chunks.append({
                "chunk_id":    _make_chunk_id(source, win, buffer_page, global_idx),
                "source":      source,
                "chapter":     cur_chapter,
                "section":     cur_section,
                "subsection":  cur_subsection,
                "page_number": buffer_page,
                "heading_path": _build_heading_path(cur_chapter, cur_section, cur_subsection),
                "chunk_type":  _detect_chunk_type(win, cur_subsection or cur_section),
                "text":        win,
                "token_count": count_tokens(win),
                # Deterministic position of this window within the document —
                # lets retrieval-time context expansion walk forward/backward
                # from a hit and reconstruct the full section in order.
                "chunk_index": global_idx
            })
            global_idx += 1
        buffer.clear()

    for page in pages:
        cur_page = page["page"]
        for block in page["blocks"]:
            text = block["text"].strip()
            if not text:
                continue

            # Skip running headers/footers and bare page numbers entirely —
            # they are neither content nor structural headings.
            if _is_running_header(text, running_headers):
                continue

            # A literal "Table of Content" listing page: suppress chapter/
            # section capture until a real UNIT/CHAPTER pattern is seen,
            # so the TOC page's own heading doesn't get adopted as the
            # chapter/section for all subsequent content.
            if _is_toc_listing_line(text):
                in_toc_listing = True
                continue
            if in_toc_listing and not _UNIT_CHAPTER_RE.match(text):
                # still inside the TOC listing block (rows like "PLANNING 14")
                continue
            if in_toc_listing and _UNIT_CHAPTER_RE.match(text):
                in_toc_listing = False  # real content resumed

            level = _is_heading_line(text, block["font_size"], block["is_bold"], body_size)
            if level == "chapter":
                flush_buffer()
                cur_chapter, cur_section, cur_subsection = text, "", ""
                buffer_page = cur_page
            elif level == "section":
                flush_buffer()
                cur_section, cur_subsection = text, ""
                buffer_page = cur_page
            elif level == "subsection":
                flush_buffer()
                cur_subsection = text
                buffer_page = cur_page
            else:
                if not buffer:
                    buffer_page = cur_page   # mark start-of-buffer page
                buffer.append(text)

    flush_buffer()
    return all_chunks


def _plain_chunk_fallback(doc: dict) -> list[dict]:
    config  = get_config()
    min_tok = config["chunking"]["min_tokens"]
    max_tok = config["chunking"]["max_tokens"]
    overlap = config["chunking"]["overlap_tokens"]
    source  = doc["source"]
    chunks  = []
    global_idx = 0

    for page in doc["pages"]:
        windows = _window_sentences(_split_sentences(page["text"]), min_tok, max_tok, overlap)
        for win in windows:
            chunks.append({
                "chunk_id":    _make_chunk_id(source, win, page["page"], global_idx),
                "source":      source,
                "chapter":     "", "section":    "",
                "subsection":  "", "page_number": page["page"],
                "heading_path": "",
                "chunk_type":  _detect_chunk_type(win),
                "text":        win,
                "token_count": count_tokens(win),
                "chunk_index": global_idx
            })
            global_idx += 1
    return chunks


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_heading_path(chapter: str, section: str, subsection: str) -> str:
    return " > ".join(p for p in [chapter, section, subsection] if p)


def _add_heading_prefix(chunk: dict) -> dict:
    path = chunk.get("heading_path", "")
    text = chunk.get("text", "")
    if path and not text.startswith(path):
        chunk["text"]        = f"{path}: {text}"
        chunk["token_count"] = count_tokens(chunk["text"])
    return chunk


def _quality_check(chunk: dict) -> dict | None:
    """
    Validate chunk size and sentence completeness before storage.

    Definitions and formulas get a relaxed minimum-token floor: per the
    "special content" requirement, these are stored as high-priority
    chunks precisely because they're often short, self-contained facts
    (a single definition sentence, a named law/theorem) — padding them
    out or dropping them for being "too short" would lose exactly the
    content a student is most likely to search for directly.
    """
    config  = get_config()
    min_tok = config["chunking"].get("quality_min_tokens", 100)
    max_tok = config["chunking"].get("quality_max_tokens", 700)

    chunk_type = chunk.get("chunk_type", "text")
    if chunk_type in ("definition", "formula"):
        min_tok = config["chunking"].get("quality_min_tokens_special", 30)

    text = chunk.get("text", "").strip()
    toks = chunk.get("token_count", count_tokens(text))

    if toks < min_tok:
        logger.debug(f"Dropped (short {toks}t, type={chunk_type}): {text[:50]!r}")
        return None

    if toks > max_tok:
        logger.debug(f"Dropped (long {toks}t): {text[:50]!r}")
        return None

    # Trim broken trailing sentence
    if text and text[-1] not in ".!?\"'":
        last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last > len(text) // 2:
            text = text[: last + 1].strip()
            chunk["text"]        = text
            chunk["token_count"] = count_tokens(text)
        if count_tokens(text) < min_tok:
            return None

    return chunk
