"""General utils for processing."""

import re


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    if isinstance(text, bytes):
        return text.decode("utf-8", "ignore")

    raise ValueError(f"Unsupported string type: {type(text)}")
