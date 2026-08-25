#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Trigger OrchestrAI pipeline builds for the resolved test batches.

For each batch (one per platform/device) this builds a pipeline plan — one group
per suite, all sharing the batch's hardware — and POSTs it to the OrchestrAI
pipeline job (buildWithParameters). It then polls the queue item until the build
starts and records its URL so the matrix job can wait on it.

Environment-specific values (pipeline URL/job, provisioning scripts, driver
sources, run settings) come from .github/orchestrai-config.yml, overridden by
repo variables. Credentials come from env.

Fails fast (exit 1) on a misconfigured run rather than acquiring scarce hardware
that cannot provision: missing config keys, missing credentials, or a batch whose
required provisioning value is unset.

Usage:
    orchestrai_trigger.py [--config .github/orchestrai-config.yml] [--dry-run]

Env:
    BATCHES_JSON            batches from orchestrai_matrix.py
    GIT_REF                 github.ref (recorded as ROCM_SYSTEMS_REF)
    GIT_SHA                 github.sha of this repo (the tests' own revision)
    MACHINES_PER_HW_GROUP   override; falls back to the config default
    ORCHESTRAI_PIPELINE_USER / ORCHESTRAI_PIPELINE_TOKEN

Outputs (to $GITHUB_OUTPUT):
    build_urls={batch_id: build url}
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import yaml

# Finite ceilings so a stalled pipeline can never hang the job (the urllib
# default is to wait forever). Generous values; the point is a bound, not an SLA.
POST_TIMEOUT = 120
POLL_TIMEOUT = 30

# Only the submit is retried: a submit that fails before the pipeline creates a
# queue item leaves nothing behind, so re-sending is safe. Once a queue item
# exists we hold that one URL, so a slow queue can never produce a duplicate
# build on scarce hardware.
SUBMIT_ATTEMPTS = 3
SUBMIT_BACKOFF = 5  # seconds, multiplied by the attempt number

QUEUE_POLL_ATTEMPTS = 60
QUEUE_POLL_INTERVAL = 10


# --- Public-repo log hygiene --------------------------------------------------
# rocm-systems is public, so every workflow log and uploaded artifact is
# world-readable. Credentials are secrets and GitHub masks those, but the
# pipeline URL MUST be a variable (secrets are redacted from job outputs, which
# would blank build_urls) and variables are never masked. So anything carrying
# internal infrastructure detail is redacted before it is printed.
_SECRET_KEY_HINTS = (
    "token",
    "password",
    "driver_source",
    "driver_sources_json",
    "url",
    "share",
)
# Deliberately NOT redacted: maas_tags. GPU architecture tags (gpuarch_rdna3 and
# friends) are not considered sensitive, and seeing which hardware a batch asked
# for is the single most useful thing in the log when a run lands on the wrong
# machine — which has happened. Hiding them would cost real debuggability to
# conceal nothing worth concealing.


def build_number(url: str) -> str:
    """Trailing build number of a pipeline build URL, for logging without the host."""
    return url.rstrip("/").rsplit("/", 1)[-1] or "?"


def redact(value):
    """Deep-copy `value` with internal coordinates replaced by a placeholder.

    Keys are matched on substrings rather than an exact list so a new
    provisioning variable is redacted by default instead of leaking until
    someone remembers to add it.
    """
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if any(h in k.lower() for h in _SECRET_KEY_HINTS):
                out[k] = "<redacted>"
            else:
                out[k] = redact(v)
        return out
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Internal coordinates are kept out of this public repo and supplied at run
    time via repo variables/secrets. Env values win over the config fallbacks."""
    pipeline = cfg.setdefault("pipeline", {})
    pipeline["url"] = os.environ.get("ORCHESTRAI_PIPELINE_URL") or pipeline.get(
        "url", ""
    )
    pipeline["job"] = os.environ.get("ORCHESTRAI_PIPELINE_JOB") or pipeline.get(
        "job", ""
    )

    prov = cfg.setdefault("provisioning", {})
    linux_driver = os.environ.get("ORCHESTRAI_LINUX_DRIVER_SOURCE")
    if linux_driver:
        prov.setdefault("linux_kernel_driver", {})["source"] = linux_driver
    linux_driver_map = os.environ.get("ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON")
    if linux_driver_map:
        prov.setdefault("linux_kernel_driver", {})["sources_json"] = linux_driver_map
    windows_driver = os.environ.get("ORCHESTRAI_WINDOWS_DRIVER_SOURCE")
    if windows_driver:
        prov.setdefault("windows_driver", {})["source"] = windows_driver
    return cfg


def validate_config(cfg: dict[str, Any], batches: dict[str, Any]) -> list[str]:
    """Return the required config keys that are missing for these batches."""
    errors = []
    run_settings = cfg.get("run_settings") or {}
    for key in ("max_duration", "max_test_case_duration", "acquire_timeout"):
        if key not in run_settings:
            errors.append(f"run_settings.{key}")
    if not (cfg.get("suites") or {}):
        errors.append("suites")

    prov = cfg.get("provisioning") or {}
    platforms = {b.get("platform") for b in batches.values()}
    if "linux" in platforms and not prov.get("linux_install_scripts"):
        errors.append("provisioning.linux_install_scripts")
    if "windows" in platforms and not prov.get("windows_install_scripts"):
        errors.append("provisioning.windows_install_scripts")
    return errors


def normalise_driver_map(raw: str) -> tuple[str, list[str]]:
    """Validate the per-release amdgpu map, returning (value, errors).

    Validated HERE, before any hardware is acquired: a malformed map would
    otherwise fail on the machine after the acquire wait. Pasting into the
    repo-variable UI commonly introduces wrapping quotes or line breaks, so both
    are repaired when what is underneath is genuinely valid JSON.
    """
    value = raw.strip()

    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        inner = value[1:-1].strip()
        try:
            json.loads(inner)
        except ValueError:
            pass
        else:
            print(
                "::warning::ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON was wrapped in "
                f"{value[0]} quotes; using the value inside them",
                file=sys.stderr,
            )
            value = inner

    try:
        json.loads(value)
    except ValueError:
        # A URL cannot contain a raw newline or tab, so dropping them is safe —
        # and only attempted when the value does not already parse.
        rejoined = re.sub(r"[\n\r\t]+", "", value)
        try:
            json.loads(rejoined)
        except ValueError:
            pass
        else:
            print(
                "::warning::ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON contained line "
                "breaks; they were removed. Set it as a single line to avoid this.",
                file=sys.stderr,
            )
            value = rejoined

    try:
        parsed = json.loads(value)
    except ValueError as exc:
        hint = ""
        if any(c in value for c in "“”‘’"):
            hint = ' — it contains smart quotes; retype it with plain " quotes'
        elif "'" in value and '"' not in value:
            hint = " — JSON requires double quotes, not single quotes"
        return value, [
            f"ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON (not valid JSON: {exc}{hint}; "
            f"{len(value)} chars, begins {value[:24]!r})"
        ]

    if not (
        isinstance(parsed, dict)
        and parsed
        and all(
            isinstance(k, str) and isinstance(v, str) and v for k, v in parsed.items()
        )
    ):
        return value, [
            "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON (expected a non-empty object of "
            '"<id>:<version_id>" -> URL)'
        ]

    # Finish repairing a value broken across lines: the newline is gone but the
    # indentation that followed it is still inside the string, so the JSON parses
    # while the URL is unusable ("curl: (3) URL rejected") much later, on the
    # machine. A URL cannot contain whitespace, so strip it there and nowhere else.
    repaired = [
        k
        for k, v in parsed.items()
        if v.startswith(("http://", "https://")) and re.search(r"\s", v)
    ]
    if repaired:
        for key in repaired:
            parsed[key] = re.sub(r"\s+", "", parsed[key])
        print(
            "::warning::ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON: removed whitespace from "
            f"the URL(s) for {', '.join(sorted(repaired))}; set the variable as a "
            "single line to avoid this",
            file=sys.stderr,
        )
        value = json.dumps(parsed, separators=(",", ":"))

    return value, []


def make_plan(
    batch: dict[str, Any],
    cfg: dict[str, Any],
    machines_per_hw_group: int,
    git_ref: str,
    git_sha: str,
) -> dict[str, Any]:
    platform = batch["platform"]
    # The repo the build came from — not necessarily the repo running this
    # workflow (a fork test run reads ROCm/rocm-systems builds).
    repo = batch["run_repo"]
    suites_cfg = cfg["suites"]
    run_settings = cfg["run_settings"]

    groups = []
    for suite_id in batch["suites"]:
        test = suites_cfg[suite_id]["test"].format(platform=platform)
        groups.append(
            {
                "id": f"suite-{suite_id}",
                # The level is the first segment of the test-library path
                # (L1-whb, L4-sys, ...), so a suite can move between levels
                # without a code change.
                "level": test.split("/", 1)[0],
                "tests": [test],
                "maas_tags": batch["tags"],
                "variables": {
                    "SUITE_ID": suite_id,
                    "TEST_PLATFORM": platform,
                    "TEST_DEVICE": batch["device"],
                    # The revision of THIS repo, so a suite that builds tests from
                    # source (e.g. hip-tests catch2) can match the build under test.
                    "ROCM_SYSTEMS_REPO": f"https://github.com/{repo}",
                    "ROCM_SYSTEMS_SHA": batch.get("sha") or git_sha,
                    "ROCM_SYSTEMS_REF": git_ref,
                    # Which TheRock build the machine was provisioned from.
                    "THEROCK_RUN_ID": batch["run_id"],
                    "THEROCK_AMDGPU_FAMILY": batch["family"],
                },
            }
        )

    return {
        "source": "rocm-systems-clr-nightly",
        "groups": groups,
        "run_settings": {
            "max_duration": run_settings["max_duration"],
            "max_test_case_duration": run_settings["max_test_case_duration"],
            "acquire_timeout": run_settings["acquire_timeout"],
            "machines_per_hw_group": machines_per_hw_group,
        },
    }


def make_builds(
    batch: dict[str, Any], cfg: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Return (builds, missing) — missing names unset required provisioning."""
    platform = batch["platform"]
    prov = cfg.get("provisioning") or {}
    missing: list[str] = []

    # Every key here reaches the install script as an upper-cased env var. These
    # four are what the artifact-based installer needs to run TheRock's
    # install_rocm_from_artifacts.py on the machine.
    build_vars = {
        "THEROCK_RUN_ID": batch["run_id"],
        "THEROCK_RUN_REPO": batch["run_repo"],
        "THEROCK_AMDGPU_FAMILY": batch["family"],
        "THEROCK_ARTIFACT_BASE_URL": batch["artifact_base_url"],
    }

    # install_rocm_from_artifacts.py ALWAYS calls the GitHub API to resolve the
    # artifact bucket, and neither it nor fetch_artifacts.py offers a way to
    # supply the bucket directly and skip the lookup. Unauthenticated it shares
    # the fleet's egress IP quota and reliably returns HTTP 429 (build 42018
    # died there). Treated as required so the run fails here rather than after
    # acquiring and rebooting a machine. Read-only public-repo scope is enough;
    # in the workflow this is ${{ github.token }}.
    github_token = os.environ.get("ORCHESTRAI_GITHUB_TOKEN", "")
    if github_token:
        build_vars["GITHUB_TOKEN"] = github_token
    else:
        missing.append("ORCHESTRAI_GITHUB_TOKEN")

    if platform == "windows":
        # Copy so the config is never mutated across batches.
        scripts = list(prov.get("windows_install_scripts") or [])
        driver = prov.get("windows_driver") or {}
        source = driver.get("source", "")
        if not source:
            missing.append("ORCHESTRAI_WINDOWS_DRIVER_SOURCE")
        build_vars["driver_source"] = source
        build_vars["driver_copy"] = driver.get("copy", "direct")
    else:
        scripts = list(prov.get("linux_install_scripts") or [])
        device_family = (cfg.get("device_families") or {}).get(batch["device"])
        if device_family not in {"radeon", "ryzen_apu"}:
            missing.append(f"device_families.{batch['device']}")
        if device_family == "radeon":
            kernel_driver = prov.get("linux_kernel_driver") or {}
            source = kernel_driver.get("source", "")
            sources_json = kernel_driver.get("sources_json", "")
            # Either form is sufficient: one URL for every Radeon host, or a
            # per-release map. The map wins on the host when both are present.
            if not source and not sources_json:
                missing.append("ORCHESTRAI_LINUX_DRIVER_SOURCE")
            if sources_json:
                sources_json, errors = normalise_driver_map(sources_json)
                missing.extend(errors)
                build_vars["driver_sources_json"] = sources_json
            if source:
                build_vars["driver_source"] = source
            # The kernel driver must be active before TheRock user-space is
            # installed. Kept device-specific so APUs keep the inbox amdgpu
            # module they are validated with.
            scripts = list(kernel_driver.get("install_scripts") or []) + scripts

    return {"install_scripts": scripts, "vars": build_vars}, missing


def auth_header(user: str, token: str) -> str:
    """Basic-auth header value, built in memory.

    SECURITY: the credentials are deliberately NOT handed to a `curl -u
    user:token` subprocess. argv is world-readable via /proc/<pid>/cmdline and
    `ps` for the lifetime of the process, so on a shared self-hosted runner any
    other local process could read the pipeline token.
    """
    return "Basic " + base64.b64encode(f"{user}:{token}".encode()).decode()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Match curl's default (no -L) so the queue URL is read from the original
    response's Location header instead of being followed."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def submit(
    plan: dict, builds: dict, platform: str, pipeline: dict, user: str, token: str
) -> str | None:
    """POST the build request; return the queue-item URL, or None on failure.

    None means no queue item was created (network error, timeout, or a
    non-redirect HTTP error), so the caller may retry without risking a duplicate.
    """
    body = urllib.parse.urlencode(
        {
            "PLAN_JSON": json.dumps(plan),
            "BUILDS_JSON": json.dumps(builds),
            "OS_IMAGE": "windows" if platform == "windows" else "ubuntu",
        }
    ).encode()

    req = urllib.request.Request(
        f"{pipeline['url'].rstrip('/')}/job/{pipeline['job']}/buildWithParameters",
        data=body,
        method="POST",
    )
    req.add_header("Authorization", auth_header(user, token))
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        opener = urllib.request.build_opener(_NoRedirect)
        with opener.open(req, timeout=POST_TIMEOUT) as resp:
            return (resp.headers.get("Location") or "").strip() or None
    except urllib.error.HTTPError as exc:
        # Redirects surface as HTTPError because they are disabled above; a 3xx
        # still carries the queue item in Location. Anything else is a failure.
        if 300 <= exc.code < 400:
            return (exc.headers.get("Location") or "").strip() or None
        return None
    except Exception:
        return None


def await_build(queue_url: str, user: str, token: str) -> str | None:
    """Poll a queue item until the pipeline assigns it an executable."""
    for _ in range(QUEUE_POLL_ATTEMPTS):
        time.sleep(QUEUE_POLL_INTERVAL)
        try:
            req = urllib.request.Request(f"{queue_url}api/json")
            req.add_header("Authorization", auth_header(user, token))
            with urllib.request.urlopen(req, timeout=POLL_TIMEOUT) as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace"))
            executable = payload.get("executable") or {}
            if executable.get("url"):
                return executable["url"]
        except Exception:
            pass
    return None


def trigger(
    plan: dict, builds: dict, platform: str, pipeline: dict, user: str, token: str
) -> str | None:
    queue_url = None
    for attempt in range(1, SUBMIT_ATTEMPTS + 1):
        queue_url = submit(plan, builds, platform, pipeline, user, token)
        if queue_url:
            break
        if attempt < SUBMIT_ATTEMPTS:
            print(
                f"  submit attempt {attempt}/{SUBMIT_ATTEMPTS} failed; retrying",
                file=sys.stderr,
            )
            time.sleep(SUBMIT_BACKOFF * attempt)
    if not queue_url:
        return None
    return await_build(queue_url, user, token)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=".github/orchestrai-config.yml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = apply_env_overrides(load_config(args.config))
    batches = json.loads(os.environ.get("BATCHES_JSON") or "{}")
    git_ref = os.environ.get("GIT_REF", "")
    git_sha = os.environ.get("GIT_SHA", "")
    user = os.environ.get("ORCHESTRAI_PIPELINE_USER", "")
    token = os.environ.get("ORCHESTRAI_PIPELINE_TOKEN", "")

    config_errors = validate_config(cfg, batches)
    if config_errors:
        print(
            "::error::orchestrai-config.yml missing required keys: "
            f"{', '.join(config_errors)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.dry_run:
        need = [f"pipeline.{k}" for k in ("url", "job") if not cfg["pipeline"].get(k)]
        if not user:
            need.append("ORCHESTRAI_PIPELINE_USER")
        if not token:
            need.append("ORCHESTRAI_PIPELINE_TOKEN")
        if need:
            print(
                f"::error::OrchestrAI not configured — missing: {', '.join(need)} "
                "(set the ORCHESTRAI_PIPELINE_URL/JOB variables and the "
                "ORCHESTRAI_PIPELINE_USER/TOKEN secrets)",
                file=sys.stderr,
            )
            sys.exit(1)

    run_settings = cfg["run_settings"]
    try:
        machines_per_hw_group = int(
            os.environ.get("MACHINES_PER_HW_GROUP", "")
            or run_settings.get("default_machines_per_hw_group", 1)
        )
    except ValueError:
        machines_per_hw_group = 1

    # Prepare every batch first and fail before any POST, so a misconfigured run
    # never acquires a scarce machine that cannot install its GPU stack and only
    # fails much later on hardware.
    prepared = []
    provisioning_missing: dict[str, list[str]] = {}
    for batch_id, batch in batches.items():
        builds, missing = make_builds(batch, cfg)
        if missing:
            provisioning_missing[batch_id] = missing
        prepared.append(
            (
                batch_id,
                batch,
                make_plan(batch, cfg, machines_per_hw_group, git_ref, git_sha),
                builds,
            )
        )

    if not args.dry_run and provisioning_missing:
        for batch_id, missing in provisioning_missing.items():
            print(
                f"::error::batch {batch_id}: unset provisioning {missing} — "
                "set the corresponding repo variable(s)/secret(s)",
                file=sys.stderr,
            )
        sys.exit(1)

    build_urls: dict[str, str] = {}
    failed: list[str] = []
    for batch_id, batch, plan, builds in prepared:
        print(
            f"Batch {batch_id}: {', '.join(batch['suites'])} "
            f"(TheRock run {batch['run_id']}, {batch['family']})",
            file=sys.stderr,
        )
        if args.dry_run:
            print(f"--- plan for {batch_id} ---")
            print(json.dumps(redact(plan), indent=2))
            print(f"--- builds for {batch_id} ---")
            print(json.dumps(redact(builds), indent=2))
            continue
        url = trigger(plan, builds, batch["platform"], cfg["pipeline"], user, token)
        if url:
            # Number only — see redact() above.
            print(f"  Build #{build_number(url)}", file=sys.stderr)
            build_urls[batch_id] = url
        else:
            failed.append(batch_id)
            print(
                f"  WARNING: trigger failed / timed out for {batch_id}", file=sys.stderr
            )

    if args.dry_run:
        return

    print(f"\nTriggered {len(build_urls)} pipeline build(s)", file=sys.stderr)

    # Emit the URLs FIRST so batches that did trigger still run downstream even
    # if we exit non-zero below.
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"build_urls={json.dumps(build_urls)}\n")

    # Surface dropped batches HERE rather than letting each downstream test job
    # discover the gap one by one as "No build URL".
    if not build_urls:
        print(
            "::error::No pipeline builds were triggered — every batch failed to "
            f"submit ({len(failed)}: {', '.join(failed)}). The pipeline was likely "
            "unreachable or rejecting builds.",
            file=sys.stderr,
        )
        sys.exit(1)
    if failed:
        # Do NOT exit non-zero: downstream jobs `needs` this one, so that would
        # skip the batches that did trigger. Their own jobs still fail, so the
        # run as a whole is not reported green.
        print(
            f"::error::{len(failed)} batch(es) failed to trigger and will report no "
            f"build URL downstream: {', '.join(failed)}. The other "
            f"{len(build_urls)} batch(es) were triggered and will run.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
