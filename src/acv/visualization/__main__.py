import sys

from .panels import build_all

_, _failed = build_all()
if _failed:
    sys.exit(f"{len(_failed)} figure(s) failed: {', '.join(_failed)}")
