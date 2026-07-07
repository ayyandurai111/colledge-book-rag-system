import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.tokenizer import count_tokens, truncate_to_limit


def test_count_tokens_basic():
    assert count_tokens("hello world") > 0

def test_count_tokens_empty():
    assert count_tokens("") == 0

def test_count_tokens_long():
    text = "word " * 200
    assert count_tokens(text) > 150

def test_truncate_no_op_short():
    text = "Short text."
    assert truncate_to_limit(text, 512) == text

def test_truncate_long_text():
    text = "This is a sentence. " * 200   # ~800 tokens
    result = truncate_to_limit(text, 512)
    assert count_tokens(result) <= 512

def test_truncate_ends_on_sentence():
    text = "First sentence. Second sentence. " * 100
    result = truncate_to_limit(text, 100)
    assert result.endswith(".")

def test_truncate_preserves_meaning():
    text = "The derivative of sin(x) is cos(x). " * 50
    result = truncate_to_limit(text, 50)
    assert len(result) > 0
