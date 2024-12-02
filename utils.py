# Utility functions for the Wisconsin Weather API
import logging
from typing import Optional, Dict, Any


def configure_logging() -> logging.Logger:
    """
    Configure and return a standard logger for the application.

    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def validate_station_id(station_id: str) -> bool:
    """
    Validate the format of a station ID.

    Args:
        station_id (str): Station identifier to validate

    Returns:
        bool: Whether the station ID is valid
    """
    # Add your specific validation logic here
    # Example: Check length, format, etc.
    return station_id is not None and len(station_id) > 0


def safe_get(
        data: Dict[str, Any],
        key: str,
        default: Optional[Any] = None
) -> Optional[Any]:
    """
    Safely retrieve a value from a dictionary.

    Args:
        data (dict): Source dictionary
        key (str): Key to retrieve
        default (Optional[Any]): Default value if key not found

    Returns:
        Optional[Any]: Retrieved value or default
    """
    return data.get(key, default)