# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Load test tiers and run the pytest suite against the staged tree."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import NoReturn


from .command import die, log, run, step
from .constants import TEST_CATEGORIES_REL, TIER_ORDER


def _yaml_to_obj(yaml_path: Path, venv_py: Path):
    """Parse a YAML file, preferring this interpreter and falling back to the venv.

    The script itself may run under a bare system/cray python without PyYAML,
    while the test venv always has it (requirements.txt pins PyYAML>=5.1), so
    shell out to the venv interpreter and exchange the result as JSON.
    """
    try:
        import yaml  # noqa: PLC0415

        return yaml.safe_load(yaml_path.read_text())
    except ImportError:
        pass
    r = subprocess.run(
        [
            str(venv_py),
            "-c",
            "import json,sys,yaml; json.dump(yaml.safe_load(open(sys.argv[1]).read()), "
            "sys.stdout)",
            str(yaml_path),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        die(f"failed to parse {yaml_path} with {venv_py}:\n{r.stderr.strip()}")
    return json.loads(r.stdout)


def _flatten(values) -> list[str]:
    """One-level flatten, so a YAML alias item (``- *anchor``) expands in place.

    Mirrors tests/pytest/conftest.py::_load_test_categories.
    """
    flat: list[str] = []
    for v in values or []:
        flat.extend(v if isinstance(v, list) else [v])
    return [str(v) for v in flat]


_TRAILING_ANY = ".*"

_REGEX_METACHARS = set(".^$*+?{}[]()|\\")


def _pattern_to_k_term(pattern: str, source: str) -> str | None:
    """Translate one YAML name regex into a pytest ``-k`` term.

    Returns None for a match-everything pattern (an empty axis in CTest terms).
    Dies on anything that is not a literal, rather than silently mis-selecting
    tests: the tier definitions have always been literal + optional ".*", and a
    real regex needs a deliberate decision here.
    """
    term = pattern
    while term.endswith(_TRAILING_ANY):
        term = term[: -len(_TRAILING_ANY)]
    if not term:
        return None
    leftover = _REGEX_METACHARS & set(term)
    if leftover:
        die(
            f"cannot translate {source} pattern {pattern!r} into a pytest -k term: "
            f"unsupported regex metacharacter(s) {''.join(sorted(leftover))}.\n"
            f"       {TEST_CATEGORIES_REL} is consumed here as literal substrings "
            "(see _pattern_to_k_term). Either keep the pattern literal or teach "
            "this script how to translate it."
        )
    return term


def _name_axis_terms(patterns, source: str, *, veto_match_all: bool) -> list[str]:
    """Translate a name axis (regex_includes / regex_excludes) into ``-k`` terms.

    A match-everything pattern collapses an include axis to a pass-through (as an
    empty ``-R`` does in CTest) and is rejected on an exclude axis, where it would
    deselect the whole suite.
    """
    terms: list[str] = []
    for pattern in patterns:
        term = _pattern_to_k_term(pattern, source)
        if term is None:
            if veto_match_all:
                die(
                    f"{TEST_CATEGORIES_REL}: {source} pattern {pattern!r} matches "
                    "every test."
                )
            return []
        terms.append(term)
    return terms


def _label_axis_terms(patterns, source: str) -> list[str]:
    """Translate a label axis (label_includes / label_excludes) into ``-m`` terms."""
    return [t for t in (_pattern_to_k_term(p, source) for p in patterns) if t]


def load_test_categories(src_dir: Path, venv_py: Path) -> dict:
    """Load the tier definitions from tests/test_categories.yaml.

    Returns ``{tier: {"includes": [...], "excludes": [...], "label_includes":
    [...], "label_excludes": [...]}}`` with every axis flattened to plain
    pytest ``-k``/``-m`` terms.

    tests/test_categories.yaml is the single source of truth for tier policy,
    shared with CTest (tests/pytest/conftest.py turns the same axes into CTest
    LABELS at generate time). Parsing it here keeps this script from drifting
    from `ctest -L <tier>`.
    """
    yaml_path = src_dir / TEST_CATEGORIES_REL
    if not yaml_path.is_file():
        die(
            f"tier definitions not found at {yaml_path}.\n"
            "       The checked-out source tree looks incomplete; re-run without "
            "--offline (or with --force-sync) to refresh it."
        )
    data = _yaml_to_obj(yaml_path, venv_py) or {}
    categories = data.get("test_categories") or {}
    missing = [t for t in TIER_ORDER if not categories.get(t)]
    if missing:
        die(f"{yaml_path} defines no {', '.join(missing)} tier(s).")

    tiers: dict = {}
    for tier in TIER_ORDER:
        cfg = categories.get(tier) or {}
        tiers[tier] = {
            "includes": _name_axis_terms(
                _flatten(cfg.get("regex_includes")),
                f"{tier}.regex_includes",
                veto_match_all=False,
            ),
            "excludes": _name_axis_terms(
                _flatten(cfg.get("regex_excludes")),
                f"{tier}.regex_excludes",
                veto_match_all=True,
            ),
            "label_includes": _label_axis_terms(
                _flatten(cfg.get("label_includes")), f"{tier}.label_includes"
            ),
            "label_excludes": _label_axis_terms(
                _flatten(cfg.get("label_excludes")), f"{tier}.label_excludes"
            ),
        }
    return tiers


def drop_label_excludes(tiers: dict, labels: list[str]) -> list[str]:
    """Stop excluding *labels* in every tier; returns the labels actually dropped.

    ``label_excludes`` in the YAML is unconditional (e.g. 'mpi' is excluded even
    by the full tier, because TheRock CI has no MPI runtime), so re-enabling a
    category has to be an explicit local override rather than a tier choice.
    """
    dropped = []
    for label in labels:
        hit = False
        for axes in tiers.values():
            if label in axes["label_excludes"]:
                axes["label_excludes"].remove(label)
                hit = True
        if hit:
            dropped.append(label)
        else:
            log(
                f"WARNING: --run-labels {label}: no tier excludes that label; "
                "nothing to re-enable."
            )
    return dropped


def tier_selection_args(tier: str, tiers: dict) -> list[str]:
    """Return the pytest -m/-k selection args for a named tier.

    The YAML axes map onto pytest the same way they map onto CTest, and both
    match on containment, so the translation is direct:
      * ``regex_includes`` (-R)  -> ``-k "(a or b)"``
      * ``regex_excludes`` (-E)  -> ``-k "not (a or b)"``
      * ``label_includes`` (-L)  -> ``-m "(a or b)"``
      * ``label_excludes`` (-LE) -> ``-m "not a and not b"``
    An empty axis is a pass-through, exactly as in CTest.

    No 'gpu' marker filter is applied, so CPU-only install-mode tests (CLI help,
    config, presets, ...) run alongside the GPU tests. The marker names come from
    pytest markers, and the name terms match the standardized (hyphenated) test
    names that conftest.py registers as extra -k keywords.
    """
    axes = tiers[tier]
    k_parts = []
    if axes["includes"]:
        k_parts.append("(" + " or ".join(axes["includes"]) + ")")
    if axes["excludes"]:
        k_parts.append("not (" + " or ".join(axes["excludes"]) + ")")
    m_parts = []
    if axes["label_includes"]:
        m_parts.append("(" + " or ".join(axes["label_includes"]) + ")")
    m_parts += [f"not {label}" for label in axes["label_excludes"]]

    args = []
    if m_parts:
        args += ["-m", " and ".join(m_parts)]
    if k_parts:
        args += ["-k", " and ".join(k_parts)]
    return args


def run_tests(
    venv_py: Path,
    pytest_dir: Path,
    rocm_dir: Path,
    env: dict,
    extra_pytest_args: str | None,
    tier: str,
    tiers: dict,
    reruns: int,
    reruns_delay: int,
    rerun_failed: int,
    workdir: Path,
    facts: dict,
    mode: str = "install",
    test_root: Path | None = None,
) -> tuple[int, Path]:
    """Run the pytest suite against either the tarball binaries or a source build.

    ``mode`` is 'install' (test the tarball's shipped binaries; ``test_root`` is the
    ROCm prefix) or 'build' (test freshly-built binaries; ``test_root`` is the build
    dir). ROCm runtime always comes from ``rocm_dir``.

    Returns ``(returncode, final_junit_path)`` where the final JUnit reflects the
    last (re)run, so callers report counts/failures from the final state.
    """
    test_env = dict(env)
    test_env["ROCM_PATH"] = str(rocm_dir)
    test_env.setdefault("ROCPROFSYS_CI", "ON")
    root = str(test_root or rocm_dir)
    if mode == "build":
        # build mode: conftest discovers the build-tree binaries via this var.
        test_env["ROCPROFSYS_BUILD_DIR"] = root
    else:
        # install mode: point the suite at the tarball prefix (bin/rocprof-sys-*).
        test_env["ROCPROFSYS_INSTALL_DIR"] = root

    # Use a private temp dir inside the work dir. This keeps per-test output out
    # of the shared /tmp AND avoids the perfetto trace_processor collision: its
    # shell is extracted to "<tempdir>/trace_processor_python_api" (a fixed name),
    # so on a shared box a copy owned by another user makes chmod fail with EPERM.
    tmpdir = workdir / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    for var in ("TMPDIR", "TMP", "TEMP"):
        test_env[var] = str(tmpdir)
    test_env.setdefault("ROCPROFSYS_TMPDIR", str(tmpdir / "rocprofsys"))
    # drop any stale perfetto shell we own so it is re-extracted under tmpdir
    stale = tmpdir / "trace_processor_python_api"
    if stale.exists():
        stale.unlink(missing_ok=True)
    log(f"Test TMPDIR: {tmpdir}")

    base = [str(venv_py), "-m", "pytest", str(pytest_dir), "-v", "-rA"]
    # A separate failed-rerun pass (--rerun-failed) needs pytest's last-failed
    # cache, so only disable the cache provider when that feature is off.
    if rerun_failed > 0:
        base += ["-o", f"cache_dir={workdir / '.pytest_cache'}"]
    else:
        base += ["-p", "no:cacheprovider"]

    if extra_pytest_args:
        # shlex, not split(): marker expressions are one argument containing spaces,
        # e.g. --pytest-args "-m 'hpc and mpi'"
        selection = shlex.split(extra_pytest_args)
    else:
        selection = tier_selection_args(tier, tiers)
        log(f"Tier '{tier}' filters: " + " ".join(shlex.quote(a) for a in selection))
    if reruns > 0:
        selection += ["--reruns", str(reruns), "--reruns-delay", str(reruns_delay)]

    junit = workdir / "pytest-results.xml"
    log(
        "Test selection: "
        + ("(custom) " + extra_pytest_args if extra_pytest_args else "tier=" + tier)
        + (f", reruns={reruns}" if reruns > 0 else "")
        + (f", rerun-failed={rerun_failed}" if rerun_failed > 0 else "")
    )
    log(f"JUnit results -> {junit}")
    cmd = base + selection + [f"--junitxml={junit}"]
    rc = run([str(c) for c in cmd], cwd=str(pytest_dir), env=test_env, check=False)

    final_junit = junit
    # Rerun only the failed tests, up to N times, each into its own JUnit artifact.
    for attempt in range(1, rerun_failed + 1):
        if rc == 0:
            break
        rerun_junit = workdir / f"pytest-rerun-{attempt}.xml"
        step(f"Re-running failed tests (attempt {attempt}/{rerun_failed})")
        cmd = base + ["--last-failed", f"--junitxml={rerun_junit}"]
        rc = run([str(c) for c in cmd], cwd=str(pytest_dir), env=test_env, check=False)
        final_junit = rerun_junit
        facts["rerun_failed_attempts"] = attempt
        if rc == 0:
            log(f"All previously-failed tests passed on rerun attempt {attempt} (flaky).")

    return rc, final_junit


def collect_failed_tests(junit: Path) -> list[tuple[str, str, str]]:
    """Return [(test_id, message, detail_text)] for failures/errors in the JUnit XML."""
    if not junit.is_file():
        return []
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(junit).getroot()
    except Exception:  # noqa: BLE001
        return []
    failed = []
    for tc in root.iter("testcase"):
        problems = tc.findall("failure") + tc.findall("error")
        if not problems:
            continue
        cls = tc.get("classname", "")
        name = tc.get("name", "")
        test_id = f"{cls}::{name}" if cls else name
        msg = (problems[0].get("message") or "").strip()
        detail = (problems[0].text or "").strip()
        failed.append((test_id, msg, detail))
    return failed


def write_failures_log(path: Path, failed: list[tuple[str, str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"Failed / errored tests: {len(failed)}\n\n")
        for test_id, msg, detail in failed:
            fh.write("=" * 72 + "\n" + test_id + "\n" + "-" * 72 + "\n")
            if msg:
                fh.write(f"message: {msg}\n")
            if detail:
                fh.write(detail + "\n")
            fh.write("\n")


def clean_previous_logs(workdir: Path, keep_stamp: str | None = None) -> None:
    """Delete log/report artifacts from previous runs (keeps rocm/, source, venv,
    and build dirs). Files matching the current run's ``keep_stamp`` are preserved."""
    patterns = [
        "detailed-*.log",
        "summary-*.log",
        "RESULT-*.md",
        "failures-*.log",
        "rocm-sanity-*.log",
        "pip-freeze-*.txt",
        "logs-*.tar.gz",
        "pytest-rerun-*.xml",
        "pytest-results.xml",
        "latest-*.txt",
    ]
    removed = 0
    for pattern in patterns:
        for f in workdir.glob(pattern):
            if keep_stamp and keep_stamp in f.name:
                continue
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    log(f"--clean: removed {removed} log/report artifact(s) from previous runs")


def parse_junit(junit: Path) -> dict | None:
    """Return aggregate pytest counts from the JUnit XML, or None if unavailable."""
    if not junit.is_file():
        return None
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(junit).getroot()
        suites = root.findall("testsuite") or ([root] if root.tag == "testsuite" else [])
        agg = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
        for s in suites:
            for k in ("tests", "failures", "errors", "skipped"):
                agg[k] += int(s.get(k, 0) or 0)
            agg["time"] += float(s.get("time", 0) or 0)
        agg["passed"] = agg["tests"] - agg["failures"] - agg["errors"] - agg["skipped"]
        return agg
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: could not parse {junit}: {exc}")
        return None
