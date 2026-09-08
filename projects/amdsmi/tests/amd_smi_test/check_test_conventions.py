#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enforce amd_smi_test C++ test conventions.

Fails (exit 1) when a test source breaks the layout, file-naming, or GTest
suite-naming rules in docs/conceptual/test-design.md. Wired in as a pre-commit
hook so a mis-placed or mis-named test — which CI's ``--gtest_filter`` would
otherwise silently skip — is caught before it lands.

To evolve the conventions, edit the CONFIGURATION block below; the checks read
from it. Run manually:

    python3 projects/amdsmi/tests/amd_smi_test/check_test_conventions.py
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration — edit these to evolve the conventions.
# --------------------------------------------------------------------------- #

TEST_ROOT = Path(__file__).resolve().parent  # projects/amdsmi/tests/amd_smi_test/
REPO_ROOT = TEST_ROOT.parents[3]
DOC = "projects/amdsmi/docs/conceptual/test-design.md"

TIERS = ("unit", "functional")  # top-level test-type directories
COMPONENTS = ("gpu", "cpu", "nic", "ifoe", "system", "wsl")  # allowed component dirs
# Functional tests group into per-feature leaf dirs (<component>/<feature>/),
# except these flat components which have no sub-features (component == feature).
FLAT_COMPONENTS = ("system",)
TEST_SUFFIX = "_test.cc"

# A GTest suite name is <Component><Type>[<Operation>]:
#   unit       -> <Component>Unit                        (no operation)
#   functional -> <Component>Functional{ReadOnly,ReadWrite}
SUITE_RE = re.compile(
    r"^(Gpu|Cpu|Nic|Ifoe|System|Wsl)(Unit|FunctionalReadOnly|FunctionalReadWrite)$"
)

# Captures the suite from TEST(Suite, Name) and TEST_F(Suite, Name).
_TEST_MACRO_RE = re.compile(r"\bTEST(?:_F)?\(\s*([A-Za-z_]\w*)\s*,")

# Declares a functional fixture, e.g. `class TestFanRead : public TestBase`.
_FIXTURE_DECL_RE = re.compile(r"^class\s+([A-Za-z_]\w*)\s*:\s*public\b", re.M)


def _pascal(component: str) -> str:
    """Component dir name -> suite prefix ('gpu' -> 'Gpu', 'ifoe' -> 'Ifoe')."""
    return "Ifoe" if component == "ifoe" else component.capitalize()


def expected_suites(tier: str, component: str) -> list[str]:
    """Suite name(s) a test under ``<tier>/<component>/`` is allowed to use."""
    prefix = _pascal(component)
    if tier == "unit":
        return [f"{prefix}Unit"]
    return [f"{prefix}FunctionalReadOnly", f"{prefix}FunctionalReadWrite"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _rel(path: Path) -> str:
    """Repo-relative path for readable messages."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _strip_comments(text: str) -> str:
    """Drop // and /* */ comments so commented-out examples (e.g. the TEST
    ENTRY TEMPLATE in main.cc) are not parsed as real tests."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _suites_in(path: Path) -> list[str]:
    """GTest suite names registered via TEST()/TEST_F() in a source file."""
    return _TEST_MACRO_RE.findall(_strip_comments(path.read_text(errors="replace")))


def _bad_pattern(path: Path, suite: str) -> str:
    return (
        f"{_rel(path)}: suite '{suite}' breaks <Component><Type>[<Operation>] "
        f"— CI --gtest_filter would skip it"
    )


# --------------------------------------------------------------------------- #
# Checks — each yields human-readable violation messages.
# --------------------------------------------------------------------------- #


def _iter_tier_sources() -> Iterator[tuple[str, str | None, Path]]:
    """Yield (tier, component, path) for every .cc under unit/ and functional/.

    ``component`` is None when the file sits loose directly under the tier dir.
    """
    for tier in TIERS:
        tier_dir = TEST_ROOT / tier
        if not tier_dir.is_dir():
            continue
        for path in sorted(tier_dir.rglob("*.cc")):
            parts = path.relative_to(tier_dir).parts
            component = parts[0] if len(parts) > 1 else None
            yield tier, component, path


def _check_layout_and_naming(tier: str, component: str | None, path: Path) -> Iterator[str]:
    """Placement (component subdir) and file-name (_test.cc) rules."""
    if component is None:
        yield (
            f"{_rel(path)}: test source must live under {tier}/<component>/..., "
            f"not directly in {tier}/"
        )
    elif component not in COMPONENTS:
        yield (f"{_rel(path)}: unknown component '{component}'; expected one of {list(COMPONENTS)}")
    elif tier == "functional" and component not in FLAT_COMPONENTS:
        # Functional tests group into per-feature leaf dirs, e.g.
        # functional/gpu/clock/frequencies_read_test.cc — parts are
        # (component, feature, file), so a missing feature leaf means len < 3.
        depth = len(path.relative_to(TEST_ROOT / tier).parts)
        if depth < 3:
            yield (
                f"{_rel(path)}: functional test must live under "
                f"{tier}/{component}/<feature>/..., not directly in {tier}/{component}/"
            )
    if not path.name.endswith(TEST_SUFFIX):
        yield (
            f"{_rel(path)}: must be named '<feature>_<operation>{TEST_SUFFIX}' "
            f"(missing '{TEST_SUFFIX}' suffix)"
        )


def _check_suites(tier: str, component: str | None, path: Path) -> Iterator[str]:
    """GTest suite-name rules for a test source file."""
    suites = _suites_in(path)

    # Pattern check is location-independent: a malformed suite is always wrong,
    # so report it even for a mis-placed file.
    for suite in suites:
        if not SUITE_RE.match(suite):
            yield _bad_pattern(path, suite)

    # Component/tier match only means something once the location is known.
    if component in COMPONENTS:
        allowed = expected_suites(tier, component)
        for suite in suites:
            if SUITE_RE.match(suite) and suite not in allowed:
                yield (
                    f"{_rel(path)}: suite '{suite}' should be {' or '.join(allowed)} "
                    f"to match its {tier}/{component}/ location"
                )


def _main_cc_tests() -> Iterator[tuple[str, str]]:
    """Yield ``(suite, body)`` for each TEST()/TEST_F() block in main.cc.

    Comments are stripped first (so disabled/example tests are ignored), and
    ``body`` spans from the macro to the start of the next TEST.
    """
    main_cc = TEST_ROOT / "main.cc"
    if not main_cc.is_file():
        return
    text = _strip_comments(main_cc.read_text(errors="replace"))
    tests = list(_TEST_MACRO_RE.finditer(text))
    for current, following in zip(tests, tests[1:] + [None]):
        body_end = following.start() if following else len(text)
        yield current.group(1), text[current.end() : body_end]


def _check_main_cc() -> Iterator[str]:
    """Functional tests register in main.cc; validate their suite names there."""
    for suite, _body in _main_cc_tests():
        if not SUITE_RE.match(suite):
            yield _bad_pattern(TEST_ROOT / "main.cc", suite)


def _functional_fixture_components() -> dict[str, str]:
    """Map each functional fixture class to its component directory.

    Reads every header under ``functional/<component>/…``; each declares one
    ``class <Fixture> : public TestBase``.
    """
    components: dict[str, str] = {}
    functional = TEST_ROOT / "functional"
    if not functional.is_dir():
        return components
    for header in functional.rglob("*.h"):
        parts = header.relative_to(functional).parts
        if len(parts) < 2:
            continue  # loose header directly under functional/, no component
        component = parts[0]
        for match in _FIXTURE_DECL_RE.finditer(header.read_text(errors="replace")):
            components[match.group(1)] = component
    return components


def _fixture_instantiated(body: str, fixtures: dict[str, str]) -> str | None:
    """Return the fixture class a TEST body instantiates (``TestFoo tst;``)."""
    for fixture in fixtures:
        if re.search(rf"\b{re.escape(fixture)}\s+\w+\s*;", body):
            return fixture
    return None


def _check_main_cc_component_match() -> Iterator[str]:
    """Flag a main.cc TEST whose suite component disagrees with the component
    directory of the fixture it runs.

    Functional tests register in main.cc, decoupled from their source directory,
    so this catches the one mismatch the per-file checks cannot see — e.g. a
    ``functional/system/`` test left under ``GpuFunctionalReadOnly``.
    """
    fixtures = _functional_fixture_components()
    main_cc = TEST_ROOT / "main.cc"
    for suite, body in _main_cc_tests():
        matched = SUITE_RE.match(suite)
        fixture = _fixture_instantiated(body, fixtures)
        if not matched or fixture is None:
            continue  # malformed suites are reported by _check_main_cc
        actual = matched.group(1)  # suite prefix: Gpu | Cpu | Nic | Ifoe | System
        expected = _pascal(fixtures[fixture])
        if actual != expected:
            yield (
                f"{_rel(main_cc)}: TEST({suite}, …) runs fixture '{fixture}' from "
                f"functional/{fixtures[fixture]}/ — suite component should be "
                f"'{expected}', not '{actual}'"
            )


def _check_stray_test_files() -> Iterator[str]:
    """A *_test.cc dropped outside unit/ and functional/ (e.g. the suite root)
    would still be compiled by aux_source_directory, so flag it."""
    for path in sorted(TEST_ROOT.rglob(f"*{TEST_SUFFIX}")):
        parts = path.relative_to(TEST_ROOT).parts
        top = parts[0] if len(parts) > 1 else None
        if top not in TIERS:
            yield (
                f"{_rel(path)}: test file must live under unit/<component>/ or "
                f"functional/<component>/ — found outside both"
            )


def collect_violations() -> list[str]:
    """Run every check and return all violation messages."""
    violations: list[str] = []
    for tier, component, path in _iter_tier_sources():
        violations += _check_layout_and_naming(tier, component, path)
        violations += _check_suites(tier, component, path)
    violations += _check_main_cc()
    violations += _check_main_cc_component_match()
    violations += _check_stray_test_files()
    return violations


def main() -> int:
    violations = collect_violations()
    if not violations:
        print("amd_smi_test conventions: OK")
        return 0
    print("amd_smi_test convention check FAILED:\n", file=sys.stderr)
    for message in violations:
        print(f"  - {message}", file=sys.stderr)
    print(f"\n{len(violations)} violation(s). See {DOC}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
