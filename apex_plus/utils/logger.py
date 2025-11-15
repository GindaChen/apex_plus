"""
Logging utility for APEX+ simulator.

Controls debug logging for time and energy calculations via environment variable.
Set APEX_DEBUG_LOG=1 to enable detailed logging to a file.
"""
import logging
import os
from pathlib import Path

# Environment variable to control debug logging
DEBUG_LOG_ENV = "APEX_DEBUG_LOG"
DEBUG_LOG_FILE_ENV = "APEX_DEBUG_LOG_FILE"

# Default log file if not specified
DEFAULT_LOG_FILE = "apex_debug.log"


def setup_debug_logger():
    """
    Set up debug logger for time and energy calculations.
    
    Returns:
        logging.Logger: Configured logger instance, or None if logging is disabled
    """
    debug_enabled = os.getenv(DEBUG_LOG_ENV, "0") == "1"
    
    if not debug_enabled:
        return None
    
    # Get log file path from environment or use default
    log_file = os.getenv(DEBUG_LOG_FILE_ENV, DEFAULT_LOG_FILE)
    log_path = Path(log_file)
    
    # Create logger
    logger = logging.getLogger("apex_debug")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_path, mode='w')  # Overwrite mode
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Log initialization message
    logger.info(f"APEX Debug Logging enabled. Log file: {log_path.absolute()}")
    
    return logger


def get_debug_logger():
    """
    Get the debug logger instance. Returns None if logging is disabled.
    
    Returns:
        logging.Logger or None: Logger instance if enabled, None otherwise
    """
    logger = logging.getLogger("apex_debug")
    if logger.handlers:
        return logger
    return None


def debug_log(message: str):
    """
    Log a debug message if debug logging is enabled.
    
    Args:
        message: Message to log
    """
    logger = get_debug_logger()
    if logger:
        logger.debug(message)


def info_log(message: str):
    """
    Log an info message if debug logging is enabled.
    
    Args:
        message: Message to log
    """
    logger = get_debug_logger()
    if logger:
        logger.info(message)


def warning_log(message: str):
    """
    Log a warning message if debug logging is enabled.
    
    Args:
        message: Message to log
    """
    logger = get_debug_logger()
    if logger:
        logger.warning(message)

