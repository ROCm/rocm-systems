#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Binary-backed gtest filter preflight for the init-pipeline (plan v11 CR-1).

A syntactically valid `test_filter` can still match zero real tests (job 216209:
exact-looking labels matched nothing, every process ran zero tests and exited 0
before READY). So before a pipeline gtest entry is admitted, resolve its filter
against the actual binary's `--gtest_list_tests` output and require exactly one
fully-qualified match for the curated gate.

This module is pure + subprocess-thin:
  * ``parse_gtest_list_tests`` — output text -> [ "Suite.Case", ... ]
  * ``gtest_filter_matches``   -> GoogleTest --gtest_filter semantics
  * ``matching_tests``         -> the FQ identities a filter selects
  * ``list_tests`` / ``preflight_filter`` — run (cached by binary identity) + match

The list-tests invocation must carry NO pipeline profile / READY-GO / rendezvous /
warmup environment.
"""

import os
import re
import subprocess

# Filters the planner treats as an unbounded whole-binary selection (rejected for
# pipeline gtest entries, v11 CR-1).
WILDCARD_FILTERS = frozenset({"", "*", "ALL", "all"})


def is_wildcard_filter(test_filter):
    """True if the filter selects the whole binary (missing / '*' / 'ALL')."""
    if test_filter is None:
        return True
    f = str(test_filter).strip()
    if f in WILDCARD_FILTERS:
        return True
    # A pure positive '*' (possibly with trailing ':') with no real token.
    tokens = [t for t in f.split(":") if t and not t.startswith("-")]
    return all(set(t) <= {"*"} for t in tokens) if tokens else True


def parse_gtest_list_tests(output):
    """Parse `--gtest_list_tests` output into fully-qualified 'Suite.Case' names.

    Handles value-/type-parameterized output: a line ending in '.' opens a suite;
    indented lines are cases (a trailing '# GetParam()...' comment is stripped).
    """
    tests = []
    suite = None
    for raw in output.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        # gtest prints a couple of banner lines (e.g. "Running main() ...");
        # suite headers are unindented and end with '.'.
        if not (line[0] == " " or line[0] == "\t"):
            stripped = line.strip()
            if stripped.endswith("."):
                suite = stripped[:-1]
            else:
                suite = None  # not a suite header (banner/other)
            continue
        if suite is None:
            continue
        case = line.strip().split("#", 1)[0].strip()
        if case:
            tests.append(f"{suite}.{case}")
    return tests


def _to_regex(pattern):
    escaped = re.sub(r'([.+^${}()|\[\]\\])', r'\\\1', pattern)
    return re.compile('^' + escaped.replace('*', '.*').replace('?', '.') + '$')


def gtest_filter_matches(name, test_filter):
    """GoogleTest --gtest_filter semantics: ':'-separated positive patterns, '-'
    starts the negative set, '*'/'?' wildcards. Empty/None matches everything."""
    if not test_filter:
        return True
    f = str(test_filter)
    pos_part, _, neg_part = f.partition("-")
    pos = [p for p in pos_part.split(":") if p]
    neg = [p for p in neg_part.split(":") if p]
    matched = True if not pos else any(_to_regex(p).match(name) for p in pos)
    if matched and neg and any(_to_regex(n).match(name) for n in neg):
        matched = False
    return matched


def matching_tests(fq_tests, test_filter):
    """FQ identities a filter selects."""
    return [t for t in fq_tests if gtest_filter_matches(t, test_filter)]


def _binary_identity(binary_path):
    """A cheap identity for caching: (path, size, mtime_ns). Falls back to path."""
    try:
        st = os.stat(binary_path)
        return (os.path.abspath(binary_path), st.st_size, st.st_mtime_ns)
    except OSError:
        return (os.path.abspath(binary_path), None, None)


def list_tests(binary_path, *, env=None, timeout=120, cache=None, runner=None):
    """Return the FQ test list for a binary via `--gtest_list_tests`, cached by
    binary identity in ``cache`` (a dict). ``runner`` overrides the subprocess call
    (for host tests). Raises RuntimeError on failure/timeout/unparsable output --
    a fatal planner error, never serial fallback (v11 CR-1)."""
    ident = _binary_identity(binary_path)
    if cache is not None and ident in cache:
        return cache[ident]

    # The preflight must NOT carry any pipeline/warmup environment.
    clean_env = dict(env if env is not None else os.environ)
    for k in ("RCCL_TEST_WARMUP_PROFILE", "RCCL_TEST_READY_GO",
              "RCCL_TEST_RENDEZVOUS_DIR", "RCCL_TEST_GO_TIMEOUT_SEC"):
        clean_env.pop(k, None)

    if runner is None:
        def runner(bp, e, to):
            proc = subprocess.run([bp, "--gtest_list_tests"], env=e,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  timeout=to, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"--gtest_list_tests exit {proc.returncode} for {bp}")
            return proc.stdout

    try:
        output = runner(binary_path, clean_env, timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"--gtest_list_tests timed out for {binary_path}") from e
    tests = parse_gtest_list_tests(output)
    if not tests:
        raise RuntimeError(f"--gtest_list_tests returned no parseable tests for {binary_path}")
    if cache is not None:
        cache[ident] = tests
    return tests


def preflight_filter(binary_path, test_filter, *, curated=True, env=None,
                     cache=None, runner=None):
    """Resolve a pipeline gtest entry's filter against the real binary.

    Returns the list of matched FQ identities. Raises ValueError if the filter is a
    wildcard, matches zero tests, or (curated gate) matches more than one.
    """
    if is_wildcard_filter(test_filter):
        raise ValueError(f"pipeline gtest filter is unbounded/wildcard: {test_filter!r}")
    fq = list_tests(binary_path, env=env, timeout=120, cache=cache, runner=runner)
    matches = matching_tests(fq, test_filter)
    if not matches:
        raise ValueError(f"test_filter {test_filter!r} matched zero tests in {binary_path}")
    if curated and len(matches) > 1:
        raise ValueError(f"test_filter {test_filter!r} matched {len(matches)} tests "
                         f"(curated gate requires exactly one): {matches[:5]}")
    return matches
