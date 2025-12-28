from __future__ import annotations
import logging
from pathlib import Path

def setup_logger(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ids")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
