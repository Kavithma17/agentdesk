import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure logging once, at the program's entry point (never on import)."""
    # A Windows console often uses cp1252, which cannot show Sinhala. Logging a Sinhala heading
    # must never crash the program, so characters the console cannot show become \u escapes.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    logging.basicConfig(level=level.upper(), format=LOG_FORMAT, force=True)
