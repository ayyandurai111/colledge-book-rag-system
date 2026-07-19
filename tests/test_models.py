import pytest

from college_rag.models import Chunk, IndexStats, SearchResult, TextBlock


def test_textblock_rejects_empty_text():
    with pytest.raises(ValueError):
        TextBlock(text="   ", source_file="x.pdf")


def test_textblock_valid():
    tb = TextBlock(text="hello", source_file="x.pdf", page_number=1, heading="Ch1")
    assert tb.text == "hello"
    assert tb.page_number == 1


def test_chunk_rejects_empty_text():
    with pytest.raises(ValueError):
        Chunk(text="", source_file="x.pdf")


def test_chunk_location_label_with_all_fields():
    c = Chunk(text="hi", source_file="book.pdf", heading="Ch1", page_number=5)
    assert c.location_label() == "book.pdf › Ch1 (page 5)"


def test_chunk_location_label_minimal():
    c = Chunk(text="hi", source_file="book.pdf")
    assert c.location_label() == "book.pdf"


def test_search_result_valid_score():
    c = Chunk(text="hi", source_file="book.pdf")
    r = SearchResult(chunk=c, score=0.95)
    assert r.score == 0.95


def test_search_result_rejects_out_of_range_score():
    c = Chunk(text="hi", source_file="book.pdf")
    with pytest.raises(ValueError):
        SearchResult(chunk=c, score=5.0)


def test_search_result_default_highlight_is_empty():
    c = Chunk(text="hi", source_file="book.pdf")
    r = SearchResult(chunk=c, score=0.9)
    assert r.highlight == ""


def test_index_stats_defaults():
    stats = IndexStats()
    assert stats.total_chunks == 0
    assert stats.source_files == []
