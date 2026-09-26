#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free unit tests for ``amdsmi_metrics_field_support``.

The module backs the default field filtering of ``amd-smi metric``: it maps a
detected ``gpu_metrics`` header version to the public fields the library can
populate, then drops the CLI output paths that version can never fill.

Two properties matter more than the size of the table:

* an unmapped version suppresses nothing, so a future metrics version degrades
  to today's output rather than losing data;
* a path is only ever dropped when every leaf under it already reads "N/A",
  so a mis-mapped path can under-report but can never hide a real value.
"""

import importlib.util
import os
import re
import sys
import unittest

try:
    from common.common import amdsmi_path
# Import-time failures in the harness are not one exception type: a checkout with
# no generated wrapper raises AttributeError out of common's enum tables. These
# tests need no harness at all, so any failure to reach it is non-fatal.
except Exception:  # pragma: no cover - harness/install unavailable
    amdsmi_path = None

# The module lives in the amd-smi CLI, which exists in two layouts:
#   * source checkout: <repo>/projects/amdsmi/amdsmi_cli
#   * installed:       <rocm>/libexec/amdsmi_cli
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_CLI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "amdsmi_cli"))
_INSTALLED_CLI_DIR = (
    os.path.join(os.path.dirname(os.path.dirname(amdsmi_path)), "libexec", "amdsmi_cli")
    if amdsmi_path
    else ""
)
_MODULE_FILENAME = "amdsmi_metrics_field_support.py"

# The C++ the version table mirrors. Present in a source checkout only.
_CPP_SOURCE_PATH = os.path.normpath(
    os.path.join(_SOURCE_CLI_DIR, "..", "rocm_smi", "src", "rocm_smi_gpu_metrics.cc")
)


def _resolve_module_path():
    for cli_dir in (_SOURCE_CLI_DIR, _INSTALLED_CLI_DIR):
        candidate = os.path.join(cli_dir, _MODULE_FILENAME) if cli_dir else ""
        if candidate and os.path.isfile(candidate):
            return candidate
    return ""


MODULE_PATH = _resolve_module_path()

# Inside a source checkout the module must exist, so a missing file is a failure
# rather than a skip. Only an installed CLI that predates the feature is skipped.
_RUNNABLE = bool(MODULE_PATH) or os.path.isdir(_SOURCE_CLI_DIR)
_SKIP_REASON = f"{_MODULE_FILENAME} not found in the installed CLI"


def _load_module():
    if not MODULE_PATH:
        raise AssertionError(
            f"{_MODULE_FILENAME} not found in {_SOURCE_CLI_DIR!r} "
            f"or {_INSTALLED_CLI_DIR or '<no installed CLI>'!r}"
        )
    spec = importlib.util.spec_from_file_location("amdsmi_metrics_field_support", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


NA = "N/A"


@unittest.skipUnless(_RUNNABLE, _SKIP_REASON)
class TestVersionFieldTable(unittest.TestCase):
    """The per-version field sets must match the C++ copy_internal_to_external_metrics()."""

    @classmethod
    def setUpClass(cls):
        cls.support = _load_module()

    def fields(self, fmt, content):
        return self.support.populatable_fields(fmt, content)

    def test_v13_carries_fields_that_v14_drops(self):
        v13 = self.fields(1, 3)
        v14 = self.fields(1, 4)
        for field in ("temperature_edge", "current_fan_speed", "average_socket_power"):
            self.assertIn(field, v13)
            self.assertNotIn(field, v14)

    def test_throttle_status_dropped_after_v15(self):
        self.assertIn("throttle_status", self.fields(1, 5))
        for content in (6, 7, 8):
            self.assertNotIn("throttle_status", self.fields(1, content))
        self.assertNotIn("throttle_status", self.fields(1, 9))

    def test_jpeg_activity_only_in_v15(self):
        self.assertNotIn("jpeg_activity", self.fields(1, 4))
        self.assertIn("jpeg_activity", self.fields(1, 5))
        for content in (6, 7, 8, 9):
            self.assertNotIn("jpeg_activity", self.fields(1, content))

    def test_vcn_activity_dropped_after_v15(self):
        for content in (3, 4, 5):
            self.assertIn("vcn_activity", self.fields(1, content))
        for content in (6, 7, 8, 9):
            self.assertNotIn("vcn_activity", self.fields(1, content))

    def test_vram_max_bandwidth_is_the_public_name_on_v17_and_v18(self):
        # v1.8 renamed the private member to mem_max_bandwidth but still copies
        # it into the public vram_max_bandwidth slot.
        self.assertNotIn("vram_max_bandwidth", self.fields(1, 6))
        for content in (7, 8, 9):
            self.assertIn("vram_max_bandwidth", self.fields(1, content))
            self.assertNotIn("mem_max_bandwidth", self.fields(1, content))

    def test_dynamic_version_field_set(self):
        for content in (9, 10, 42):
            dynamic = self.fields(1, content)
            self.assertIsNotNone(dynamic)
            for present in (
                "temperature_hbm_stacks",
                "current_socclks_mid",
                "current_uclk_aid",
                "xgmi_link_status",
            ):
                self.assertIn(present, dynamic)
            for absent in ("temperature_edge", "jpeg_activity", "num_partition"):
                self.assertNotIn(absent, dynamic)

    def test_xcp_stats_sub_fields_are_version_scoped(self):
        v16 = self.fields(1, 6)
        for present in ("xcp_stats.gfx_busy_inst", "xcp_stats.jpeg_busy", "xcp_stats.vcn_busy"):
            self.assertIn(present, v16)
        self.assertNotIn("xcp_stats.temperature_xcd", v16)
        self.assertNotIn("xcp_stats.gfx_busy_inst", self.fields(1, 5))
        self.assertIn("xcp_stats.temperature_xcd", self.fields(1, 8))
        self.assertIn("xcp_stats.temperature_xcd", self.fields(1, 9))

    def test_apu_versions_differ(self):
        v24 = self.fields(2, 4)
        v30 = self.fields(3, 0)
        self.assertNotEqual(v24, v30)
        for only_v24 in ("throttle_status", "current_gfxclk", "current_uclk", "current_dclk0"):
            self.assertIn(only_v24, v24)
            self.assertNotIn(only_v24, v30)
        self.assertIn("average_gfx_activity", v30)

    def test_unknown_version_returns_none(self):
        for version in ((9, 9), (0, 0), (4, 0), (2, 5), (1, "N/A")):
            self.assertIsNone(self.fields(*version))

    def test_future_content_revision_reuses_the_dynamic_bound(self):
        # amdgpu_metrics_factory() routes every (1, >=9) header to the dynamic
        # reader, so a newer content revision is mapped rather than unknown.
        self.assertEqual(self.fields(1, 99999), self.fields(1, 9))


@unittest.skipUnless(_RUNNABLE, _SKIP_REASON)
class TestSuppressionSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.support = _load_module()

    def suppressed(self, fmt, content):
        return self.support.build_suppression_set(
            {"structure_size": 512, "format_revision": fmt, "content_revision": content}
        )

    def test_unknown_version_suppresses_nothing(self):
        self.assertEqual(self.suppressed(9, 9), frozenset())

    def test_missing_header_suppresses_nothing(self):
        for header in (None, {}, "N/A", {"format_revision": "N/A", "content_revision": "N/A"}):
            self.assertEqual(self.support.build_suppression_set(header), frozenset())

    def test_v13_suppresses_v16_only_paths(self):
        v13 = self.suppressed(1, 3)
        self.assertIn(("usage", "jpeg_activity"), v13)
        self.assertIn(("throttle", "prochot_accumulated"), v13)
        self.assertNotIn(("usage", "vcn_activity"), v13)
        self.assertNotIn(("temperature", "edge"), v13)

    def test_v19_suppresses_removed_legacy_paths(self):
        v19 = self.suppressed(1, 9)
        for path in (
            ("usage", "jpeg_activity"),
            ("usage", "vcn_activity"),
            ("usage", "mm_activity"),
            ("power", "throttle_status"),
        ):
            self.assertIn(path, v19)

    def test_paths_with_a_non_metrics_source_are_never_mapped(self):
        """Only paths sourced solely from the metrics blobs may be suppressed; anything
        metric.py can also fill from hwmon, sysfs or another API must stay."""
        for path in (
            ("temperature", "edge"),
            ("temperature", "hotspot"),
            ("temperature", "mem"),
            ("fan", "speed"),
            ("fan", "max"),
            ("fan", "rpm"),
            ("fan", "usage"),
            ("power", "gfx_voltage"),
            ("power", "soc_voltage"),
            ("power", "mem_voltage"),
            ("pcie", "replay_count"),
            ("clock", "mem_0"),
            ("clock", "socclk_0"),
            ("usage", "vcn_busy"),
        ):
            self.assertNotIn(path, self.support._CLI_PATH_REQUIRED_SOURCES)
            for fmt, content in ((1, 0), (1, 3), (1, 6), (1, 9), (2, 4), (3, 0)):
                self.assertNotIn(path, self.suppressed(fmt, content))

    def test_unmapped_paths_are_never_suppressed(self):
        for fmt, content in ((1, 0), (1, 3), (1, 9), (3, 0)):
            suppressed = self.suppressed(fmt, content)
            for path in (
                ("mem_usage", "total_vram"),
                ("ecc", "total_correctable_count"),
                ("clock", "fclk_0"),
                ("energy", "total_energy_consumption"),
            ):
                self.assertNotIn(path, suppressed)


@unittest.skipUnless(_RUNNABLE, _SKIP_REASON)
class TestEverySourceIsRequired(unittest.TestCase):
    """Sources listed for a CLI path are required together, not alternatives.

    The multi-source entries are the derived throttle rows: the activity is a
    percentage of a residency accumulator over ``accumulation_counter`` and the
    status is that percentage tested against zero, so a version carrying only
    one of the two can never fill either form. Reading the sources as
    alternatives ("suppress only when none is present") left those rows printing
    "N/A" under the flag.
    """

    DERIVED_FORMS = ("_violation_activity", "_violation_status")

    # Rows a version cannot fill because its accumulator is absent while
    # accumulation_counter is present. Pinned so the predicate cannot silently
    # flip back to the alternatives reading.
    _GFX_LIMIT_ROWS = (
        "gfx_clk_below_host_limit",
        "gfx_clk_below_host_limit_power",
        "gfx_clk_below_host_limit_thermal",
        "total_gfx_clk_below_host_limit",
        "low_utilization",
    )
    SUPPRESSED_DERIVED_ROWS = {
        (1, 6): frozenset(_GFX_LIMIT_ROWS),
        (1, 7): frozenset(_GFX_LIMIT_ROWS) - {"gfx_clk_below_host_limit"},
        (1, 8): frozenset({"gfx_clk_below_host_limit"}),
        (1, 9): frozenset({"gfx_clk_below_host_limit"}),
    }

    @classmethod
    def setUpClass(cls):
        cls.support = _load_module()

    def suppressed(self, fmt, content):
        return self.support.build_suppression_set(
            {"format_revision": fmt, "content_revision": content}
        )

    def test_multi_source_entries_are_all_accumulator_over_counter(self):
        # The pinned counts below only mean what they say while this holds.
        for path, sources in self.support._CLI_PATH_REQUIRED_SOURCES.items():
            if len(sources) > 1:
                self.assertEqual(sources[-1], "accumulation_counter", path)
                self.assertEqual(len(sources), 2, path)

    def test_only_the_derived_throttle_forms_need_the_counter(self):
        for row, source in self.support._THROTTLE_SOURCES.items():
            self.assertEqual(
                self.support._CLI_PATH_REQUIRED_SOURCES[("throttle", f"{row}_accumulated")],
                (source,),
            )
            for form in self.DERIVED_FORMS:
                self.assertEqual(
                    self.support._CLI_PATH_REQUIRED_SOURCES[("throttle", f"{row}{form}")],
                    (source, "accumulation_counter"),
                )

    def test_derived_rows_missing_an_accumulator_are_suppressed(self):
        for version, expected in sorted(self.SUPPRESSED_DERIVED_ROWS.items()):
            for form in self.DERIVED_FORMS:
                with self.subTest(version=version, form=form):
                    # These versions carry the counter, so only the required-
                    # source reading can suppress these rows.
                    self.assertIn("accumulation_counter", self.support.populatable_fields(*version))
                    actual = frozenset(
                        key[: -len(form)]
                        for section, key in self.suppressed(*version)
                        if section == "throttle" and key.endswith(form)
                    )
                    self.assertEqual(actual, expected)

    def test_a_derived_row_never_outlives_its_accumulator(self):
        suffix = "_accumulated"
        for version in sorted(self.support.known_versions()) + [(1, 9), (1, 12)]:
            with self.subTest(version=version):
                suppressed = self.suppressed(*version)
                for section, key in suppressed:
                    if section == "throttle" and key.endswith(suffix):
                        row = key[: -len(suffix)]
                        for form in self.DERIVED_FORMS:
                            self.assertIn(("throttle", f"{row}{form}"), suppressed)


@unittest.skipUnless(_RUNNABLE, _SKIP_REASON)
class TestFilterUnsupported(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.support = _load_module()

    def test_drops_suppressed_na_leaves(self):
        values = {
            "gpu": 0,
            "usage": {"gfx_activity": 42, "jpeg_activity": [NA, NA], "vcn_activity": [NA]},
        }
        filtered = self.support.filter_unsupported(
            values, frozenset({("usage", "jpeg_activity"), ("usage", "vcn_activity")})
        )
        self.assertEqual(filtered, {"gpu": 0, "usage": {"gfx_activity": 42}})

    def test_indexed_clock_slots_are_matched_literally(self):
        # Suppression is by exact (section, key); a family name matches nothing.
        na_clock = {"clk": NA, "min_clk": NA, "max_clk": NA, "clk_locked": NA, "deep_sleep": NA}
        values = {"clock": {"gfx_0": dict(na_clock), "gfx_1": dict(na_clock)}}
        self.assertEqual(
            self.support.filter_unsupported(values, frozenset({("clock", "gfx")})), values
        )

    def test_collapses_section_left_empty(self):
        values = {"gpu": 1, "usage": {"mm_activity": NA, "jpeg_activity": NA}}
        suppressed = frozenset({("usage", "mm_activity"), ("usage", "jpeg_activity")})
        self.assertEqual(self.support.filter_unsupported(values, suppressed), {"gpu": 1})

    def test_protected_section_survives_whole_instead_of_collapsing(self):
        values = {"gpu": 1, "usage": {"mm_activity": NA, "jpeg_activity": NA}}
        suppressed = frozenset({("usage", "mm_activity"), ("usage", "jpeg_activity")})
        self.assertEqual(
            self.support.filter_unsupported(values, suppressed, frozenset({"usage"})), values
        )

    def test_protected_section_is_still_filtered_when_something_survives(self):
        values = {"usage": {"gfx_activity": 42, "jpeg_activity": NA}}
        suppressed = frozenset({("usage", "jpeg_activity")})
        self.assertEqual(
            self.support.filter_unsupported(values, suppressed, frozenset({"usage"})),
            {"usage": {"gfx_activity": 42}},
        )

    def test_keeps_section_that_was_already_empty(self):
        values = {"ecc_blocks": {}}
        self.assertEqual(
            self.support.filter_unsupported(values, frozenset({("usage", "mm_activity")})), values
        )

    def test_recognizes_human_readable_na_list_string(self):
        values = {
            "usage": {"jpeg_activity": "[N/A, N/A, N/A]", "vcn_activity": "[N/A, 5 %]"},
            "temperature": {"hbm_stacks": "[]"},
        }
        suppressed = frozenset(
            {("usage", "jpeg_activity"), ("usage", "vcn_activity"), ("temperature", "hbm_stacks")}
        )
        filtered = self.support.filter_unsupported(values, suppressed)
        self.assertNotIn("jpeg_activity", filtered["usage"])
        self.assertEqual(filtered["usage"]["vcn_activity"], "[N/A, 5 %]")
        self.assertEqual(filtered["temperature"]["hbm_stacks"], "[]")

    def test_does_not_mutate_input(self):
        values = {"usage": {"gfx_activity": 42, "jpeg_activity": [NA, NA]}}
        before = {"usage": {"gfx_activity": 42, "jpeg_activity": [NA, NA]}}
        self.support.filter_unsupported(values, frozenset({("usage", "jpeg_activity")}))
        self.assertEqual(values, before)

    def test_empty_suppression_set_is_identity(self):
        values = {"gpu": 0, "usage": {"gfx_activity": NA}, "fan": {"speed": NA}}
        self.assertEqual(self.support.filter_unsupported(values, frozenset()), values)

    def test_never_drops_a_real_value(self):
        """Safety invariant: every mapped path suppressed for every known version,
        with a real value in place, must survive the filter."""
        real_values = (
            7,
            0,
            "ENABLED",
            [1, 2, 3],
            [NA, 4],
            "[N/A, 4 %]",
            {"value": 3, "unit": "%"},
            {"xcp_0": [NA, 9]},
        )
        versions = list(self.support.known_versions()) + [(1, 9), (1, 12)]
        for fmt, content in versions:
            suppressed = self.support.build_suppression_set(
                {"format_revision": fmt, "content_revision": content}
            )
            for index, real in enumerate(real_values):
                values = {}
                for section, key in suppressed:
                    values.setdefault(section, {})[key] = real
                filtered = self.support.filter_unsupported(values, suppressed)
                self.assertEqual(
                    filtered,
                    values,
                    f"version {fmt}.{content} dropped real value #{index}: {real!r}",
                )


@unittest.skipUnless(_RUNNABLE, _SKIP_REASON)
class TestVersionTableMatchesCppSource(unittest.TestCase):
    """Drift guard: the hand-written table must track the C++ it mirrors.

    Derives each version's public field set from the per-version
    ``copy_internal_to_external_metrics()`` in ``rocm_smi_gpu_metrics.cc`` and
    diffs it against ``_VERSION_FIELDS``. Skipped where the C++ source is not
    shipped, such as an installed package.
    """

    # Class name -> the (format_revision, content_revision) it reads.
    CLASS_VERSIONS = {f"GpuMetricsBase_v1{content}_t": (1, content) for content in range(9)}
    DYNAMIC_CLASS = "GpuMetricsBaseDynamic_t"
    # APU versions (2, 4) and (3, 0) share one class and are told apart by a
    # branch inside it, so CLASS_VERSIONS cannot map them;
    # test_apu_versions_match_their_branches covers them instead.
    APU_CLASS = "ApuMetricsBase_v30_t"

    # common_header is the version stamp itself; apu_metrics is the internal
    # source struct, not a public member. xcp_stats sub-fields are assigned in a
    # copy loop the extractor cannot see, so the bare name is dropped here and
    # the sub-fields are excluded from every comparison by _comparable().
    IGNORED_ROOTS = frozenset({"common_header", "apu_metrics", "xcp_stats"})

    _FUNCTION_RE = re.compile(
        r"^AMGpuMetricsPublicLatestTupl_t\s+(\w+)::copy_internal_to_external_metrics\(\)\s*\{",
        re.MULTILINE,
    )
    # A body ends at its own closing brace in column 0. Without this the last
    # class in the file would absorb every later assignment in the file.
    _FUNCTION_END_RE = re.compile(r"^\}", re.MULTILINE)
    # v1.x classes copy into a local named metrics_public_init; the dynamic
    # reader copies into one named out.
    _ASSIGNMENT_RE = re.compile(r"\b(?:metrics_public_init|out)\.([A-Za-z_][A-Za-z0-9_.]*)")
    _LINE_COMMENT_RE = re.compile(r"//[^\n]*")
    # ApuMetricsBase_v30_t serves both APU versions from one function: a v2.4
    # branch and an else branch for v3.0. Comparing the merged body would let a
    # deleted v3.0 field pass, so each branch is extracted and compared alone.
    _APU_V24_BRANCH_RE = re.compile(
        r"if\s*\(\s*get_gpu_metrics_version_used\(\)\s*==\s*[\w:]*kApuMetricV24\s*\)\s*\{"
    )
    # Branch braces sit one level in; the bodies themselves are deeper.
    _APU_ELSE_RE = re.compile(r"^  \} else \{", re.MULTILINE)
    _APU_BRANCH_END_RE = re.compile(r"^  \}", re.MULTILINE)

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(_CPP_SOURCE_PATH):
            raise unittest.SkipTest(f"C++ source not available at {_CPP_SOURCE_PATH}")
        cls.support = _load_module()
        with open(_CPP_SOURCE_PATH, encoding="utf-8", errors="replace") as source:
            cls.bodies = cls._derive_bodies(source.read())
        cls.derived = {name: cls._fields_in(body) for name, body in cls.bodies.items()}

    @classmethod
    def _derive_bodies(cls, text):
        functions = list(cls._FUNCTION_RE.finditer(text))
        bodies = {}
        for index, match in enumerate(functions):
            limit = functions[index + 1].start() if index + 1 < len(functions) else len(text)
            closing = cls._FUNCTION_END_RE.search(text, match.end(), limit)
            end = closing.end() if closing else limit
            bodies[match.group(1)] = cls._LINE_COMMENT_RE.sub("", text[match.start() : end])
        return bodies

    @classmethod
    def _fields_in(cls, body):
        return {
            name
            for name in cls._ASSIGNMENT_RE.findall(body)
            if name.split(".")[0] not in cls.IGNORED_ROOTS
        }

    @classmethod
    def _apu_branch_fields(cls):
        """(v2.4 fields, v3.0 fields), or None if the branches cannot be located."""
        body = cls.bodies.get(cls.APU_CLASS, "")
        opening = cls._APU_V24_BRANCH_RE.search(body)
        if opening is None:
            return None
        otherwise = cls._APU_ELSE_RE.search(body, opening.end())
        if otherwise is None:
            return None
        closing = cls._APU_BRANCH_END_RE.search(body, otherwise.end())
        if closing is None:
            return None
        return (
            cls._fields_in(body[opening.end() : otherwise.start()]),
            cls._fields_in(body[otherwise.end() : closing.start()]),
        )

    @staticmethod
    def _comparable(fields):
        # xcp_stats.* is deliberately outside the drift guard: the C++ assigns
        # those sub-fields inside a per-partition copy loop, so a line-based
        # extractor cannot attribute them to a version. Guarding them needs a
        # branch-aware extractor. Until then 18 of the 49 mapped CLI paths are
        # unguarded and rest on the explicit xcp_stats tests above.
        return {name for name in fields if not name.startswith("xcp_stats.")}

    def _assert_matches(self, class_name, table_fields):
        self.assertIn(
            class_name, self.derived, f"{class_name}::copy_internal_to_external_metrics() not found"
        )
        expected = self._comparable(self.derived[class_name])
        actual = self._comparable(table_fields)
        self.assertEqual(
            actual,
            expected,
            f"{class_name} drifted from the table | "
            f"only in table: {sorted(actual - expected)} | "
            f"only in C++: {sorted(expected - actual)}",
        )

    def test_every_mapped_class_is_present_in_the_source(self):
        missing = sorted(
            name
            for name in list(self.CLASS_VERSIONS) + [self.DYNAMIC_CLASS, self.APU_CLASS]
            if name not in self.derived
        )
        self.assertEqual(missing, [], f"classes renamed or removed in C++: {missing}")

    def test_explicit_versions_match_the_cpp_source(self):
        for class_name, version in sorted(self.CLASS_VERSIONS.items(), key=lambda item: item[1]):
            with self.subTest(version=version):
                table_fields = self.support.populatable_fields(*version)
                self.assertIsNotNone(table_fields, f"{version} is missing from _VERSION_FIELDS")
                self._assert_matches(class_name, table_fields)

    def test_dynamic_bound_matches_the_cpp_source(self):
        self._assert_matches(
            self.DYNAMIC_CLASS,
            self.support.populatable_fields(1, self.support._DYNAMIC_MIN_CONTENT_REVISION),
        )

    def test_apu_versions_match_their_branches(self):
        branches = self._apu_branch_fields()
        self.assertIsNotNone(
            branches,
            f"could not split {self.APU_CLASS}::copy_internal_to_external_metrics() into its "
            "kApuMetricV24 and else branches; the branch shape changed",
        )
        for version, branch in zip(((2, 4), (3, 0)), branches):
            with self.subTest(version=version):
                expected = self._comparable(branch)
                actual = self._comparable(self.support.populatable_fields(*version))
                self.assertEqual(
                    actual,
                    expected,
                    f"{version} drifted from its {self.APU_CLASS} branch | "
                    f"only in table: {sorted(actual - expected)} | "
                    f"only in C++: {sorted(expected - actual)}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
