import logging
from pathlib import Path


def setup_logger(log_file: str) -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("IDS")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
