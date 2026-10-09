#!/usr/bin/env python3
"""Guard that Makefile header-dep tracking uses -MM (NCCL PR #1806).

NCCL 2.30.7 switched include tracking from -M to -MM so system headers are
omitted from generated .d files. src/device/Makefile (and the diagnostics
copy) then realpath-filters those paths down to the tree; listing libc / HIP
/ compiler headers made that loop excessively slow on some filesystems.

This is a source scan, not a compile. Nothing in RCCL CI runs ctest, so the
copy that gates is test/host/run_host_tests.sh `guards` (folded into `run`).
"""

from __future__ import annotations

import os
import re
import unittest

RCCL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(RCCL_ROOT, "src")

# Longest-match first so -MMD is not read as -MM or -M, and -MD / -MM are
# not read as -M. -march and friends fail the trailing boundary.
DEP_FLAG_RE = re.compile(r"(?<!\S)(-MMD|-MM|-MD|-MF|-MG|-MP|-MT|-MQ|-M)(?=\s|$)")

# -MM / -MMD omit system headers. -M / -MD include them (the #1806 bug).
USER_ONLY = frozenset({"-MM", "-MMD"})
SYSTEM_INCLUSIVE = frozenset({"-M", "-MD"})

# Files that carry the NCCL #1806 recipes (plus the RCCL diagnostics copy).
REQUIRED = (
    ("Makefile", ("d_tmp",), 2),
    (os.path.join("device", "Makefile"), ("DEPENDS.cu", "DEPENDS.cc"), 2),
    (os.path.join("diagnostics", "device", "Makefile"), ("DEPENDS.cu",), 1),
)


def _join_continued_lines(text):
    """Collapse backslash continuations into (first lineno, logical line)."""
    logical = []
    buf = []
    start = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if start is None:
            start = lineno
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            buf.append(stripped[:-1])
            continue
        buf.append(stripped)
        logical.append((start, " ".join(buf)))
        buf = []
        start = None
    if buf:
        logical.append((start, " ".join(buf)))
    return logical


def _dep_flags(line):
    return tuple(DEP_FLAG_RE.findall(line))


def _is_depgen_line(line):
    """True for a compiler dep-tracking recipe, not a later sed of .d.tmp."""
    if not line or line.lstrip().startswith("#"):
        return False
    flags = _dep_flags(line)
    if not flags:
        return False
    return (
        ".d.tmp" in line
        or re.search(r"DEPENDS\.(cu|cc)\s*=", line) is not None
    )


def depgen_sites(text):
    """Return (lineno, line, flags) for each header-dep recipe in `text`."""
    sites = []
    for lineno, line in _join_continued_lines(text):
        if _is_depgen_line(line):
            sites.append((lineno, line, _dep_flags(line)))
    return sites


def site_errors(flags):
    """Human-readable problems for one recipe's -M* flags, or []."""
    have = set(flags)
    errors = []
    bad = have & SYSTEM_INCLUSIVE
    if bad:
        errors.append("uses system-header flag %s (want -MM or -MMD)" %
                      ", ".join(sorted(bad)))
    if not (have & USER_ONLY):
        errors.append("missing -MM / -MMD")
    return errors


class TestMakefileDepFlags(unittest.TestCase):
    def test_parser_accepts_dash_mm(self):
        sites = depgen_sites(
            "\t@$(CXX) -I. $(CXXFLAGS) -MM $< > $(@:%.o=%.d.tmp)\n"
        )
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0][2], ("-MM",))
        self.assertEqual(site_errors(sites[0][2]), [])

    def test_parser_rejects_dash_m(self):
        sites = depgen_sites(
            "\t@$(CXX) -I. $(CXXFLAGS) -M $< > $(@:%.o=%.d.tmp)\n"
        )
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0][2], ("-M",))
        self.assertTrue(site_errors(sites[0][2]))

    def test_parser_rejects_dash_md(self):
        sites = depgen_sites("DEPENDS.cu = $(NVCC) $(NVCUFLAGS) -MD -dc $1\n")
        self.assertEqual(len(sites), 1)
        self.assertTrue(site_errors(sites[0][2]))

    def test_parser_ignores_sed_of_d_tmp(self):
        text = (
            "\t@sed \"0,/^.*:/s//x:/\" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)\n"
            "\t$(CXX) -c $< -o $@\n"
        )
        self.assertEqual(depgen_sites(text), [])

    def test_parser_does_not_treat_march_as_dep_flag(self):
        sites = depgen_sites(
            "\t$(CXX) -march=native -MM $< > foo.d.tmp\n"
        )
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0][2], ("-MM",))

    def test_parser_reports_physical_lineno(self):
        # Three continued lines plus the recipe: logical index would be 2,
        # physical line of the -MM rule is 4.
        text = (
            "foo = \\\n"
            "  bar \\\n"
            "  baz\n"
            "\t@$(CXX) -MM $< > foo.d.tmp\n"
        )
        sites = depgen_sites(text)
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0][0], 4)

    def test_parser_sees_mm_after_abutting_backslash(self):
        # Join with a space so $(CXXFLAGS)\ + -MM does not fuse into
        # $(CXXFLAGS)-MM, which DEP_FLAG_RE would miss.
        text = (
            "DEPENDS.cc = $(CXX) $(CXXFLAGS)\\\n"
            "-MM -c $1\n"
        )
        sites = depgen_sites(text)
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0][0], 1)
        self.assertEqual(sites[0][2], ("-MM",))

    def test_src_makefiles_use_user_only_deps(self):
        failures = []
        for rel, kinds, min_sites in REQUIRED:
            path = os.path.join(SRC_DIR, rel)
            self.assertTrue(os.path.isfile(path), "missing %s" % path)
            with open(path) as f:
                text = f.read()
            sites = depgen_sites(text)
            if len(sites) < min_sites:
                failures.append(
                    "%s: expected at least %d depgen site(s), found %d"
                    % (rel, min_sites, len(sites))
                )
            for kind in kinds:
                if kind == "d_tmp":
                    if not any(".d.tmp" in line for _, line, _ in sites):
                        failures.append("%s: no .d.tmp -MM recipe" % rel)
                    continue
                if not any(kind in line for _, line, _ in sites):
                    failures.append("%s: no %s recipe" % (rel, kind))
            for lineno, line, flags in sites:
                errs = site_errors(flags)
                if errs:
                    failures.append(
                        "%s:%d: %s\n    %s"
                        % (rel, lineno, "; ".join(errs), line.strip())
                    )
        self.assertFalse(failures, "NCCL PR #1806 -MM guard failed:\n" +
                         "\n".join(failures))


if __name__ == "__main__":
    unittest.main(verbosity=2)
