#!/usr/bin/env python3
"""
GitHub Actions Data Fetcher
----------------------------
Fetches workflow run data from the GitHub Actions API for CI analysis.
Supports date-range filtering, month bucketing, and job-level queries.

Authentication:
    - GH_TOKEN environment variable (preferred)
    - Falls back to `gh auth token` CLI command
"""

import calendar
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Workflow registry: maps logical keys to exact workflow filenames on disk.
WORKFLOW_REGISTRY = {
    "therock_ci": {
        "filename": "therock-ci.yml",
        "label": "TheRock CI",
        "category": "primary",
    },
    "therock_nightly": {
        "filename": "therock-ci-nightly.yml",
        "label": "TheRock CI Nightly",
        "category": "nightly",
    },
    "amdsmi": {
        "filename": "amdsmi-build.yml",
        "label": "AMDSMI CI",
        "category": "per-project",
    },
    "media_libs": {
        "filename": "media-libs-ci.yml",
        "label": "Media Libs CI",
        "category": "per-project",
    },
    "aqlprofile": {
        "filename": "aqlprofile-continuous_integration.yml",
        "label": "AqlProfile CI",
        "category": "per-project",
    },
    "rccl": {
        "filename": "therock-rccl-ci.yml",
        "label": "RCCL CI",
        "category": "per-project",
    },
    "rocprof_systems": {
        "filename": "rocprofiler-systems-continuous-integration.yml",
        "label": "rocprofiler-systems CI",
        "category": "per-project",
    },
    "rocprof_sdk": {
        "filename": "rocprofiler-sdk-continuous_integration.yml",
        "label": "rocprofiler-sdk CI",
        "category": "per-project",
    },
}


def _resolve_token() -> str:
    """Resolve GitHub token from env var or gh CLI."""
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if token:
        return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    raise SystemExit(
        "Error: No GitHub token found.\n"
        "Set GH_TOKEN environment variable or authenticate with: gh auth login"
    )


class CIDataFetcher:
    """Fetches workflow run data from the GitHub Actions REST API."""

    API_URL = "https://api.github.com"

    def __init__(self, repo: str = "ROCm/rocm-systems", token: Optional[str] = None):
        self.repo = repo
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token or _resolve_token()}",
                "Accept": "application/vnd.github+json",
            }
        )
        self._workflow_id_cache: dict[str, int] = {}

    def _get_with_retries(
        self,
        url: str,
        params: Optional[dict] = None,
        retries: int = 3,
        backoff: int = 2,
        timeout: int = 30,
    ) -> Optional[requests.Response]:
        """GET with retry, exponential backoff, and rate-limit handling."""
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                if response.status_code == 200:
                    return response
                if (
                    response.status_code == 403
                    and response.headers.get("X-RateLimit-Remaining") == "0"
                ):
                    reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                    sleep_seconds = max(1, reset_time - int(time.time()) + 1)
                    logger.warning(f"Rate limited. Sleeping {sleep_seconds}s...")
                    time.sleep(sleep_seconds)
                    continue
                if response.status_code in {403, 429, 500, 502, 503, 504}:
                    logger.warning(
                        f"Retryable {response.status_code} on attempt {attempt + 1}"
                    )
                else:
                    logger.error(
                        f"HTTP {response.status_code} for {url}: {response.text[:200]}"
                    )
                    return None
            except requests.RequestException as e:
                logger.warning(f"Request failed attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(backoff**attempt)
        logger.error(f"Max retries reached for {url}")
        return None

    def _get_json(self, url: str, params: Optional[dict] = None) -> dict:
        """GET returning a single JSON object."""
        response = self._get_with_retries(url, params=params)
        return response.json() if response else {}

    def _get_paginated(
        self, url: str, params: Optional[dict] = None, max_pages: int = 10
    ) -> list[dict]:
        """GET with pagination, collecting items from the list endpoint."""
        results = []
        page = 0
        while url and page < max_pages:
            response = self._get_with_retries(url, params=params)
            if not response:
                break
            data = response.json()
            # GitHub Actions list endpoints wrap results in a key
            if isinstance(data, dict):
                for key in ("workflow_runs", "jobs", "workflows"):
                    if key in data:
                        results.extend(data[key])
                        break
                else:
                    results.extend(data.get("items", [data]))
            elif isinstance(data, list):
                results.extend(data)
            url = response.links.get("next", {}).get("url")
            params = None  # params are in the next URL already
            page += 1
        return results

    def _get_workflow_id(self, filename: str) -> Optional[int]:
        """Look up workflow ID by filename, caching the full list on first call."""
        if not self._workflow_id_cache:
            url = f"{self.API_URL}/repos/{self.repo}/actions/workflows"
            workflows = self._get_paginated(url, params={"per_page": 100})
            for wf in workflows:
                path = wf.get("path", "")
                name = path.split("/")[-1] if "/" in path else path
                self._workflow_id_cache[name] = wf["id"]
        return self._workflow_id_cache.get(filename)

    def get_workflow_runs(
        self,
        filename: str,
        created_range: Optional[str] = None,
        event: Optional[str] = None,
        status: Optional[str] = None,
        per_page: int = 100,
        max_pages: int = 5,
    ) -> list[dict]:
        """Fetch workflow runs with optional filters.

        Args:
            filename: Workflow filename (e.g. 'therock-ci.yml')
            created_range: GitHub date range (e.g. '2026-04-01..2026-04-30')
            event: Filter by event type (e.g. 'pull_request', 'schedule')
            status: Filter by conclusion (e.g. 'completed')
            per_page: Results per page
            max_pages: Max pages to fetch
        """
        wf_id = self._get_workflow_id(filename)
        if not wf_id:
            logger.warning(f"Workflow not found: {filename}")
            return []

        url = f"{self.API_URL}/repos/{self.repo}/actions/workflows/{wf_id}/runs"
        params = {"per_page": per_page}
        if created_range:
            params["created"] = created_range
        if event:
            params["event"] = event
        if status:
            params["status"] = status

        return self._get_paginated(url, params=params, max_pages=max_pages)

    def get_run_jobs(self, run_id: int) -> list[dict]:
        """Fetch all jobs for a workflow run."""
        url = f"{self.API_URL}/repos/{self.repo}/actions/runs/{run_id}/jobs"
        return self._get_paginated(url, params={"per_page": 100}, max_pages=3)

    @staticmethod
    def compute_month_ranges(
        months: int = 3, reference_date: Optional[datetime] = None
    ) -> list[tuple[str, str, str]]:
        """Compute (month_key, start_date, end_date) tuples for the last N months.

        Returns list ordered from oldest to newest.
        """
        ref = reference_date or datetime.now()
        ranges = []
        for i in range(months - 1, -1, -1):
            # Walk back i months
            year = ref.year
            month = ref.month - i
            while month <= 0:
                month += 12
                year -= 1
            last_day = calendar.monthrange(year, month)[1]
            # For current month, cap at today
            if year == ref.year and month == ref.month:
                last_day = min(last_day, ref.day)
            month_key = f"{year}-{month:02d}"
            start = f"{year}-{month:02d}-01"
            end = f"{year}-{month:02d}-{last_day:02d}"
            ranges.append((month_key, start, end))
        return ranges

    def fetch_all(
        self, months: int = 3, reference_date: Optional[datetime] = None
    ) -> dict[str, dict[str, list[dict]]]:
        """Fetch runs for all registered workflows, bucketed by month.

        Returns:
            {workflow_key: {month_key: [run_dicts, ...]}}
        """
        month_ranges = self.compute_month_ranges(months, reference_date)
        result: dict[str, dict[str, list[dict]]] = {}

        for wf_key, wf_info in WORKFLOW_REGISTRY.items():
            filename = wf_info["filename"]
            label = wf_info["label"]
            result[wf_key] = {}

            for month_key, start, end in month_ranges:
                created_range = f"{start}..{end}"
                logger.info(f"Fetching {label} runs for {month_key} ({created_range})")
                runs = self.get_workflow_runs(
                    filename, created_range=created_range, status="completed"
                )
                result[wf_key][month_key] = runs
                logger.info(f"  -> {len(runs)} runs")

        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fetcher = CIDataFetcher()
    data = fetcher.fetch_all(months=1)
    for wf_key, months in data.items():
        label = WORKFLOW_REGISTRY[wf_key]["label"]
        for month_key, runs in months.items():
            success = sum(1 for r in runs if r.get("conclusion") == "success")
            print(f"{label} ({month_key}): {len(runs)} runs, {success} success")
