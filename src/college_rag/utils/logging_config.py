"""
logging_config.py
------------------
A single place to configure logging for the whole package. Entry points
(apps) should call `setup_logging()` once at startup.
"""
import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger("college_rag")
    if root.handlers:
        # Idempotent: already configured (e.g. Streamlit re-runs the script)
        root.setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)
