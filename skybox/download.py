"""Tiny caching HTTP fetcher built on the standard library."""

from __future__ import annotations

import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

USER_AGENT = "skybox-generator/0.1 (python-urllib; personal use)"


class DownloadError(RuntimeError):
    pass


def fetch(url: str, cache_dir: str, cache_key: str, max_retries: int = 3) -> str:
    """Download `url` to `cache_dir/cache_key`, returning the local path.

    Already-cached files are returned without hitting the network.
    """
    path = os.path.join(cache_dir, cache_key)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(path, "wb") as fh:
                fh.write(data)
            return path
        except Exception as exc:  # noqa: BLE001 - we retry then surface a clean error
            last_err = exc
            time.sleep(1.5 * (2 ** attempt))

    raise DownloadError(f"Failed to download {url} after {max_retries} tries: {last_err}")


def fetch_many(
    jobs: list[tuple[str, str]],
    cache_dir: str,
    workers: int = 8,
) -> list[str | None]:
    """Fetch many (url, cache_key) pairs concurrently.

    Returns one local path per job, or None when a job ultimately failed.
    """
    results: list[str | None] = [None] * len(jobs)

    def _run(idx: int) -> None:
        url, key = jobs[idx]
        try:
            results[idx] = fetch(url, cache_dir, key)
        except DownloadError:
            results[idx] = None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_run, range(len(jobs))))

    return results