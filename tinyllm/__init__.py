"""Kindling — a small language model written from scratch in PyTorch."""

import sys


def utf8_console():
    """On Windows, redirected stdin/stdout default to cp1252, which garbles or crashes on Romanian diacritics."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
