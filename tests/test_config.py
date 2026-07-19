import pytest

from college_rag.config import Config


def test_default_config_is_valid():
    Config().validate()  # should not raise


def test_max_must_exceed_min():
    cfg = Config(min_chunk_chars=500, max_chunk_chars=200)
    with pytest.raises(ValueError):
        cfg.validate()


def test_min_chunk_chars_must_be_positive():
    cfg = Config(min_chunk_chars=0, max_chunk_chars=100)
    with pytest.raises(ValueError):
        cfg.validate()


def test_breakpoint_percentile_out_of_range():
    cfg = Config(breakpoint_percentile=150)
    with pytest.raises(ValueError):
        cfg.validate()


def test_top_k_must_be_positive():
    cfg = Config(default_top_k=0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_highlight_sentences_must_be_positive():
    cfg = Config(highlight_sentences=0)
    with pytest.raises(ValueError):
        cfg.validate()
