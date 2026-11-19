#!/usr/bin/env python3
"""Send one madengine A/B leg's record to CoCo, which stores it for its madengine page.

The record is the madengine_result.json test_madengine.py leaves in its work
dir. It is POSTed gzipped to <RCCL_COCO_URL>/api/data/write/madengine-ab with
the API token in RCCL_COCO_WEB_TOKEN. With either unset this does nothing,
so the workflow is unchanged until both are configured.

A record sent twice replaces the first, so a failed send is retried. The
step that runs this carries continue-on-error: CoCo being down must not fail
a leg, and a leg that never reaches CoCo still shows there from GitHub's run
list.

Usage:
  python3 publish_madengine_result.py <work dir>/madengine_result.json
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROUTE = "/api/data/write/madengine-ab"
ATTEMPTS = 3
RETRY_DELAY_SECONDS = 10
TIMEOUT_SECONDS = 60


def build_request(url: str, token: str, record: dict) -> urllib.request.Request:
    """The POST of *record*: strict JSON, gzipped, under the record's own schema version."""
    return urllib.request.Request(
        url.rstrip("/") + ROUTE,
        data=gzip.compress(json.dumps(record, allow_nan=False).encode()),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
            "X-Coco-Payload-Version": str(record["schema_version"]),
        },
    )


def retryable(status: int) -> bool:
    """Whether CoCo's answer is worth asking again: a 4xx says the record itself was refused."""
    return status == 429 or status >= 500


def publish(url: str, token: str, record: dict) -> bool:
    """Send *record*, retrying what may pass on another try; whether CoCo stored it."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(build_request(url, token, record),
                                        timeout=TIMEOUT_SECONDS) as resp:
                print(f"CoCo stored the leg: {resp.read().decode(errors='replace')}")
                return True
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            print(f"::warning::CoCo answered {exc.code} (attempt {attempt}): {detail}")
            if not retryable(exc.code):
                return False
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"::warning::Could not reach CoCo (attempt {attempt}): {exc}")
        if attempt < ATTEMPTS:
            time.sleep(RETRY_DELAY_SECONDS * attempt)
    return False


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    url = os.environ.get("RCCL_COCO_URL", "")
    token = os.environ.get("RCCL_COCO_WEB_TOKEN", "")
    if not url or not token:
        print("RCCL_COCO_URL or RCCL_COCO_WEB_TOKEN is not set; not publishing to CoCo.")
        return 0
    path = Path(argv[1])
    if not path.is_file():
        print(f"::notice::No CoCo record at {path}: the leg stopped before writing one.")
        return 0
    return 0 if publish(url, token, json.loads(path.read_text())) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
