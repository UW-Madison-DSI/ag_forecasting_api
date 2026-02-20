"""
Shared async HTTP client utilities.

Centralises session creation and retry logic so every API service module
obtains sessions and retries through the same code path.
"""

import asyncio
import logging

import aiohttp

from api.config.constants import (
    HTTP_CONN_LIMIT,
    HTTP_DNS_TTL,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


def create_session() -> aiohttp.ClientSession:
    """
    Create a configured ``aiohttp.ClientSession``.

    Usage::

        async with create_session() as session:
            ...
    """
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(
            limit=HTTP_CONN_LIMIT,
            ttl_dns_cache=HTTP_DNS_TTL,
        ),
        timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS),
    )


async def get_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    params: dict,
    max_retries: int = HTTP_MAX_RETRIES,
) -> dict | None:
    """
    Execute a GET request with exponential back-off on rate-limit (429) responses.

    Args:
        session:     An open ``aiohttp.ClientSession``.
        url:         Target URL.
        params:      Query-string parameters.
        max_retries: Maximum number of attempts before giving up.

    Returns:
        Parsed JSON body (dict) on success, or ``None`` after all retries fail.
    """
    for attempt in range(max_retries):
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                if response.status == 429:
                    wait = 2 ** attempt
                    logger.warning("Rate-limited (429) on %s – waiting %ds", url, wait)
                    await asyncio.sleep(wait)
                else:
                    logger.warning("HTTP %s from %s (attempt %d)", response.status, url, attempt + 1)
                    await asyncio.sleep(1)
        except aiohttp.ClientError as exc:
            logger.error("Client error on %s (attempt %d): %s", url, attempt + 1, exc)
            await asyncio.sleep(1)

    logger.error("Giving up on %s after %d attempts", url, max_retries)
    return None