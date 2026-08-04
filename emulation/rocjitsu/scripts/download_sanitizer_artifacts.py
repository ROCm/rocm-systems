#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Download prebuilt waitcheck and ConSan artifacts from GitHub Actions.

The ``rocjitsu-sanitizer-artifacts`` workflow publishes a bundle containing
``bin/rj_waitcheck`` and ``lib/librocjitsu_dbi_hooks.so`` for every build of
the sanitizer integration branch.  This script resolves a workflow run
(``--run latest`` by default), downloads its artifact, verifies the recorded
SHA-256 digests, and unpacks a ready-to-use tree.

GitHub does not serve Actions artifacts anonymously, so a token with
``actions:read`` scope is required even though the repository is public.  The
token is taken from ``--token``, then ``$GITHUB_TOKEN``, then ``$GH_TOKEN``,
and finally from ``gh auth token``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile

DEFAULT_REPOSITORY = 'ROCm/rocm-systems'
DEFAULT_WORKFLOW = 'rocjitsu-sanitizer-artifacts.yml'
DEFAULT_BRANCH = 'shared/rocjitsu/sanitizers'
ARTIFACT_PREFIX = 'rocjitsu-sanitizers'
API_ROOT = 'https://api.github.com'
API_VERSION = '2022-11-28'
# Artifacts are large enough that a short read timeout produces spurious
# failures on slow links; the API calls themselves are small.
API_TIMEOUT_SECONDS = 30
DOWNLOAD_TIMEOUT_SECONDS = 600
EXECUTABLE_BITS = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH


class DownloadError(RuntimeError):
    """A user-facing failure that should not print a traceback."""


def resolve_token(explicit: str | None) -> str:
    """Return an API token, preferring explicit input over the environment."""
    if explicit:
        return explicit
    for variable in ('GITHUB_TOKEN', 'GH_TOKEN'):
        value = os.environ.get(variable)
        if value:
            return value
    gh = shutil.which('gh')
    if gh:
        try:
            completed = subprocess.run(
                [gh, 'auth', 'token'],
                capture_output=True,
                text=True,
                check=True,
                timeout=API_TIMEOUT_SECONDS,
            )
        except (subprocess.SubprocessError, OSError):
            pass
        else:
            token = completed.stdout.strip()
            if token:
                return token
    raise DownloadError(
        'no GitHub token available. GitHub requires authentication to '
        'download Actions artifacts, even from public repositories.\n'
        'Provide one with --token, set GITHUB_TOKEN, or run `gh auth login`.'
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surface redirects as errors so the caller controls the second request."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_error(error: urllib.error.HTTPError, url: str) -> DownloadError:
    """Translate an HTTP failure into an actionable message."""
    detail = error.read().decode('utf-8', 'replace').strip()
    if error.code in (401, 403):
        return DownloadError(
            f'GitHub rejected the request ({error.code}). The token needs '
            f'`actions:read` scope for this repository.\n{detail}'
        )
    if error.code == 404:
        return DownloadError(
            f'not found ({url}). Check --repo, --workflow, and --run.\n{detail}'
        )
    if error.code == 410:
        return DownloadError(
            f'the artifact has expired and is no longer downloadable.\n{detail}'
        )
    return DownloadError(f'GitHub request failed ({error.code}): {detail}')


def api_headers(token: str | None, accept: str) -> dict[str, str]:
    headers = {'Accept': accept, 'User-Agent': 'rocjitsu-download-sanitizer-artifacts'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
        headers['X-GitHub-Api-Version'] = API_VERSION
    return headers


def request(url: str, token: str | None, *, timeout: int, accept: str) -> bytes:
    """Perform a GET and return the response body, authenticating when given a token."""
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=api_headers(token, accept)),
            timeout=timeout,
        ) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise http_error(error, url) from error
    except urllib.error.URLError as error:
        raise DownloadError(f'could not reach GitHub: {error.reason}') from error


def get_json(url: str, token: str) -> dict:
    body = request(
        url, token, timeout=API_TIMEOUT_SECONDS, accept='application/vnd.github+json'
    )
    return json.loads(body)


def find_latest_run(repository: str, workflow: str, branch: str, token: str) -> dict:
    """Return the most recent successful workflow run for a branch."""
    query = urllib.parse.urlencode(
        {
            'branch': branch,
            'status': 'success',
            'per_page': '1',
            'exclude_pull_requests': 'true',
        }
    )
    url = f'{API_ROOT}/repos/{repository}/actions/workflows/{workflow}/runs?{query}'
    payload = get_json(url, token)
    runs = payload.get('workflow_runs') or []
    if not runs:
        raise DownloadError(
            f'no successful `{workflow}` run found on branch `{branch}`.\n'
            'Trigger the workflow, or pass an explicit --run ID.'
        )
    return runs[0]


def get_run(repository: str, run_id: str, token: str) -> dict:
    return get_json(f'{API_ROOT}/repos/{repository}/actions/runs/{run_id}', token)


def list_artifacts(repository: str, run_id: int, token: str) -> list[dict]:
    url = f'{API_ROOT}/repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100'
    return get_json(url, token).get('artifacts') or []


def select_artifact(artifacts: list[dict], name: str | None) -> dict:
    """Pick the requested artifact, or the sole sanitizer bundle."""
    live = [artifact for artifact in artifacts if not artifact.get('expired')]
    if not live:
        raise DownloadError(
            'the run has no unexpired artifacts. Artifacts are retained for a '
            'limited window; pick a newer run.'
        )
    if name:
        for artifact in live:
            if artifact['name'] == name:
                return artifact
        available = ', '.join(sorted(artifact['name'] for artifact in live))
        raise DownloadError(f'artifact `{name}` not found. Available: {available}')

    matching = [
        artifact for artifact in live if artifact['name'].startswith(ARTIFACT_PREFIX)
    ]
    if len(matching) == 1:
        return matching[0]
    if not matching:
        available = ', '.join(sorted(artifact['name'] for artifact in live))
        raise DownloadError(
            f'no `{ARTIFACT_PREFIX}*` artifact in this run. Available: {available}'
        )
    available = ', '.join(sorted(artifact['name'] for artifact in matching))
    raise DownloadError(f'several bundles matched; pass --artifact. Found: {available}')


def download_artifact(repository: str, artifact: dict, token: str) -> bytes:
    """Fetch an artifact zip, handling the redirect to blob storage by hand.

    The API answers with a 302 to a pre-signed storage URL. That URL carries
    its own credentials, and the storage backend rejects the request outright
    if a stray ``Authorization`` header rides along -- which is exactly what
    urllib would do if it followed the redirect for us.
    """
    url = f'{API_ROOT}/repos/{repository}/actions/artifacts/{artifact["id"]}/zip'
    opener = urllib.request.build_opener(_NoRedirect)
    headers = api_headers(token, 'application/vnd.github+json')
    try:
        with opener.open(
            urllib.request.Request(url, headers=headers),
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code not in (301, 302, 303, 307, 308):
            raise http_error(error, url) from error
        location = error.headers.get('Location')
        error.close()
        if not location:
            raise DownloadError(
                'GitHub redirected the download without a Location header'
            )
        if urllib.parse.urlparse(location).scheme != 'https':
            raise DownloadError(
                f'refusing to follow a non-HTTPS redirect to {location}'
            )
    except urllib.error.URLError as error:
        raise DownloadError(f'could not reach GitHub: {error.reason}') from error

    return request(location, None, timeout=DOWNLOAD_TIMEOUT_SECONDS, accept='*/*')


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> list[Path]:
    """Extract an archive, rejecting entries that escape the destination."""
    root = destination.resolve()
    extracted: list[Path] = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        target = (root / info.filename).resolve()
        if not target.is_relative_to(root):
            raise DownloadError(
                f'refusing to extract outside destination: {info.filename}'
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, open(target, 'wb') as sink:
            shutil.copyfileobj(source, sink)
        extracted.append(target)
    return extracted


def verify_digests(destination: Path) -> int:
    """Check extracted payload files against the bundle's sha256sums.txt."""
    checksums = destination / 'sha256sums.txt'
    if not checksums.is_file():
        print(
            'warning: bundle has no sha256sums.txt; skipping verification',
            file=sys.stderr,
        )
        return 0

    verified = 0
    for line in checksums.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        expected, _, relative = line.partition('  ')
        relative = relative.strip()
        if not expected or not relative:
            raise DownloadError(f'malformed sha256sums.txt entry: {line}')
        target = destination / relative
        if not target.is_file():
            raise DownloadError(f'bundle is missing a checksummed file: {relative}')
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != expected:
            raise DownloadError(
                f'checksum mismatch for {relative}\n  expected {expected}\n  actual   {digest}'
            )
        verified += 1
    if not verified:
        raise DownloadError('sha256sums.txt contained no entries')
    return verified


def mark_executable(destination: Path) -> None:
    """Restore the executable bit that the artifact zip does not preserve."""
    for target in list((destination / 'bin').glob('*')) + list(
        (destination / 'lib').glob('*.so')
    ):
        if target.is_file():
            target.chmod(target.stat().st_mode | EXECUTABLE_BITS)


def describe_run(run: dict) -> str:
    return (
        f'run {run["id"]} (#{run.get("run_number", "?")}) '
        f'{run.get("head_branch", "?")}@{(run.get("head_sha") or "")[:12]} '
        f'{run.get("status", "?")}/{run.get("conclusion", "?")} '
        f'{run.get("created_at", "?")}'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--run',
        default='latest',
        help='workflow run ID, or "latest" for the newest successful build (default: latest)',
    )
    parser.add_argument(
        '--branch',
        default=DEFAULT_BRANCH,
        help=f'branch to resolve "latest" against (default: {DEFAULT_BRANCH})',
    )
    parser.add_argument(
        '--repo',
        default=DEFAULT_REPOSITORY,
        help=f'owner/name (default: {DEFAULT_REPOSITORY})',
    )
    parser.add_argument(
        '--workflow',
        default=DEFAULT_WORKFLOW,
        help=f'workflow file name or ID (default: {DEFAULT_WORKFLOW})',
    )
    parser.add_argument(
        '--artifact',
        help='artifact name; inferred when the run publishes a single bundle',
    )
    parser.add_argument(
        '--dest',
        type=Path,
        default=Path('rocjitsu-sanitizers'),
        help='directory to unpack into (default: ./rocjitsu-sanitizers)',
    )
    parser.add_argument(
        '--keep-zip', action='store_true', help='also save the downloaded artifact zip'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='overwrite a non-empty destination directory',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='list the resolved run and its artifacts, then exit',
    )
    parser.add_argument(
        '--token', help='GitHub token (default: $GITHUB_TOKEN, $GH_TOKEN, gh)'
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = resolve_token(args.token)

    if args.run == 'latest':
        run = find_latest_run(args.repo, args.workflow, args.branch, token)
    else:
        if not args.run.isdigit():
            raise DownloadError(
                f'--run must be a numeric run ID or "latest", got `{args.run}`'
            )
        run = get_run(args.repo, args.run, token)

    print(f'resolved {describe_run(run)}')
    print(f'  {run.get("html_url", "")}')

    artifacts = list_artifacts(args.repo, run['id'], token)
    if args.list:
        if not artifacts:
            print('  (no artifacts)')
        for artifact in artifacts:
            state = 'expired' if artifact.get('expired') else 'available'
            size_mib = artifact.get('size_in_bytes', 0) / (1024 * 1024)
            print(f'  {artifact["name"]}  {size_mib:.1f} MiB  {state}')
        return 0

    if run.get('conclusion') != 'success':
        print(
            f'warning: run conclusion is `{run.get("conclusion")}`; '
            'artifacts may be incomplete',
            file=sys.stderr,
        )

    artifact = select_artifact(artifacts, args.artifact)

    # Reject an occupied destination before spending the download.
    destination = args.dest
    if destination.exists() and any(destination.iterdir()) and not args.force:
        raise DownloadError(f'{destination} is not empty; pass --force to overwrite')

    size_mib = artifact.get('size_in_bytes', 0) / (1024 * 1024)
    print(f'downloading {artifact["name"]} ({size_mib:.1f} MiB)')
    payload = download_artifact(args.repo, artifact, token)
    destination.mkdir(parents=True, exist_ok=True)

    if args.keep_zip:
        zip_path = destination.with_suffix('.zip')
        zip_path.write_bytes(payload)
        print(f'saved {zip_path}')

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            extracted = safe_extract(archive, destination)
    except zipfile.BadZipFile as error:
        raise DownloadError(
            f'downloaded artifact is not a valid zip: {error}'
        ) from error

    verified = verify_digests(destination)
    mark_executable(destination)

    print(f'extracted {len(extracted)} files to {destination}')
    if verified:
        print(f'verified {verified} SHA-256 digests')

    hook = destination / 'lib' / 'librocjitsu_dbi_hooks.so'
    waitcheck = destination / 'bin' / 'rj_waitcheck'
    if hook.is_file() and waitcheck.is_file():
        print('\nRun the combined waitcheck + ConSan hook with:')
        print(f'  env HSA_TOOLS_DISABLE_REGISTER=1 \\')
        print(f'      HSA_TOOLS_LIB="{hook.resolve()}" \\')
        print(f'      ./application')
        print('\nInspect a saved code object with:')
        print(f'  "{waitcheck.resolve()}" path/to/kernel.hsaco')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except DownloadError as error:
        print(f'error: {error}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
