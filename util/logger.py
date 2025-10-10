import logging
import os
from pathlib import Path

def get_logger(output_dir: Path, name: str) -> logging.Logger:
    """
    Initializes and returns a logger that writes to both a file and the console.

    Args:
        output_dir: The directory where the log file will be saved.
        name: The name of the logger.

    Returns:
        A configured logging.Logger instance.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # --- [MODIFIED] Check if handlers are already present to avoid duplication ---
    if logger.hasHandlers():
        return logger

    log_file = output_dir / f"{name}.log"

    # --- File Handler (now in append mode) ---
    # The 'a' mode ensures that if the file exists, new logs are appended.
    # If it doesn't exist, it will be created.
    file_handler = logging.FileHandler(log_file, mode='a')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # --- Console Handler ---
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s') # Keep console output clean
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
