"""Shared CI utilities for GitHub Actions workflow scripts."""

import fnmatch
import logging
import os
import subprocess
import time
from typing import Callable, Iterable, Mapping, Tuple, Type, TypeVar

F = TypeVar("F", bound=Callable[..., object])

logging.basicConfig(level=logging.INFO)


def retry(
    max_attempts: int,
    delay_seconds: float,
    exceptions: Tuple[Type[BaseException], ...],
) -> Callable[[F], F]:
    """Retry decorator with exponential backoff."""

    def decorator(func: F) -> F:
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(
                        f"Exception {str(e)} thrown when attempting to run, "
                        f"attempt {attempt} of {max_attempts}"
                    )
                    attempt += 1
                    if attempt < max_attempts:
                        backoff = delay_seconds * (2 ** (attempt - 1))
                        time.sleep(backoff)
            return func(*args, **kwargs)

        return wrapper

    return decorator


@retry(max_attempts=3, delay_seconds=2, exceptions=(TimeoutError,))
def get_modified_paths(base_ref: str) -> set[str]:
    """Returns paths of files changed relative to the base reference."""
    result = subprocess.run(
        ["git", "diff", "--name-only", base_ref],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return set(result.stdout.splitlines())


def matches_paths(changed_files: Iterable[str], patterns: Iterable[str]) -> bool:
    """Returns True if any changed file matches any of the fnmatch patterns."""
    for f in changed_files:
        for pattern in patterns:
            if fnmatch.fnmatch(f, pattern):
                return True
    return False


def set_github_output(outputs: Mapping[str, str]):
    """Writes key=value pairs to GITHUB_OUTPUT (or prints if not in CI)."""
    logging.info(f"Setting github output:\n{dict(outputs)}")
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not output_file:
        for k, v in outputs.items():
            print(f"{k}={v}")
        return
    with open(output_file, "a") as f:
        for k, v in outputs.items():
            f.write(f"{k}={v}\n")
