#!/usr/bin/env python3
"""Unit tests for check_default_behavior_changes.py.

Every trigger condition is covered by a sample diff, per AICOMRCCL-1904 AC-3.
Stdlib only, so the same `python3 -m unittest` run works in CI and locally.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_default_behavior_changes import (  # noqa: E402
    Declaration,
    Detection,
    NO_ENTRY,
    TRIGGER_DEFAULT,
    TRIGGER_NEW,
    TRIGGER_REFERENCE,
    build_report,
    coverage_gap,
    detect,
    is_covered,
    parse_declarations,
    parse_diff,
    parse_exemptions,
    parse_md_entries,
    suggested_row,
)


def make_diff(path, hunk_header, lines):
    """Assemble a unified diff for one file from raw hunk body lines."""
    return "\n".join(
        ["diff --git a/%s b/%s" % (path, path), "--- a/%s" % path, "+++ b/%s" % path]
        + [hunk_header]
        + lines
    )


def registry_of(*decls):
    return {decl.accessor: decl for decl in decls}


PARAM_FOO = Declaration("NCCL_FOO", "ncclParamFoo", "0", "src/enqueue.cc", 10, 10)
PARAM_BAR = Declaration("RCCL_BAR", "rcclParamBar", "1", "src/init.cc", 20, 20)


class ParseDeclarationsTest(unittest.TestCase):
    def test_quoted_macros_get_their_prefix(self):
        text = 'NCCL_PARAM(Foo, "FOO", 0);\nRCCL_PARAM(Bar, "BAR", 1);\n'
        decls = parse_declarations(text, "src/x.cc")
        self.assertEqual(
            [(d.env, d.accessor, d.default) for d in decls],
            [("NCCL_FOO", "ncclParamFoo", "0"), ("RCCL_BAR", "rcclParamBar", "1")],
        )

    def test_alias_macro_yields_both_names(self):
        decls = parse_declarations('RCCL_PARAM_NCCL_ALIAS(Baz, "BAZ", 2);', "src/x.cc")
        self.assertEqual([d.env for d in decls], ["RCCL_BAZ", "NCCL_BAZ"])
        self.assertEqual({d.accessor for d in decls}, {"rcclParamBaz"})

    def test_define_macro_key_is_the_full_name(self):
        text = (
            "DEFINE_NCCL_PARAM(ncclParamDebugLevel, ncclDebugLogLevel, NCCL_DEBUG,\n"
            "                  NCCL_LOG_NONE, NCCL_PARAM_FLAG_NONE, parser, "
            '"desc");\n'
        )
        decls = parse_declarations(text, "src/debug.cc")
        self.assertEqual(len(decls), 1)
        self.assertEqual(decls[0].env, "NCCL_DEBUG")
        self.assertEqual(decls[0].default, "NCCL_LOG_NONE")

    def test_declare_macro_is_not_a_definition(self):
        self.assertEqual(parse_declarations("RCCL_PARAM_DECLARE(Foo);", "src/x.h"), [])

    def test_multiline_declaration_spans_lines(self):
        text = 'RCCL_PARAM(UseAmdSmiLib, "USE_AMD_SMI_LIB",\n           1);\n'
        decls = parse_declarations(text, "src/x.cc")
        self.assertEqual(decls[0].default, "1")
        self.assertEqual((decls[0].line, decls[0].end_line), (1, 2))

    def test_expression_default_is_kept_verbatim(self):
        decls = parse_declarations(
            'NCCL_PARAM(Pool, "POOL", 1LL << 30);', "src/allocator.cc"
        )
        self.assertEqual(decls[0].default, "1LL << 30")

    def test_truncated_declaration_is_skipped(self):
        self.assertEqual(parse_declarations('NCCL_PARAM(Foo, "FOO",', "src/x.cc"), [])


class NewParamTest(unittest.TestCase):
    def test_new_param_reports_default_and_line(self):
        diff = make_diff(
            "src/enqueue.cc",
            "@@ -40,0 +41,1 @@",
            ['+NCCL_PARAM(NewKnob, "NEW_KNOB", 0);'],
        )
        found = detect(parse_diff(diff), {})
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].env, "NCCL_NEW_KNOB")
        self.assertEqual(found[0].trigger, TRIGGER_NEW)
        self.assertEqual((found[0].path, found[0].line), ("src/enqueue.cc", 41))
        self.assertEqual(found[0].new, "0")

    def test_alias_param_reports_both_names(self):
        diff = make_diff(
            "src/init.cc",
            "@@ -1,0 +2,1 @@",
            ['+RCCL_PARAM_NCCL_ALIAS(Knob, "KNOB", 1);'],
        )
        self.assertEqual(
            sorted(d.env for d in detect(parse_diff(diff), {})),
            ["NCCL_KNOB", "RCCL_KNOB"],
        )


class DefaultChangeTest(unittest.TestCase):
    def test_default_change_reports_old_and_new(self):
        diff = make_diff(
            "src/init.cc",
            "@@ -12,1 +12,1 @@",
            ['-NCCL_PARAM(Knob, "KNOB", 1);', '+NCCL_PARAM(Knob, "KNOB", 2);'],
        )
        found = detect(parse_diff(diff), {})
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].trigger, TRIGGER_DEFAULT)
        self.assertEqual((found[0].old, found[0].new), ("1", "2"))

    def test_multiline_declaration_with_only_the_value_changed(self):
        """The macro name sits in context; only the default line is added."""
        diff = make_diff(
            "src/misc/signals.cc",
            "@@ -61,2 +61,2 @@",
            [
                ' RCCL_PARAM(EnableSignalHandler, "ENABLE_SIGNALHANDLER",',
                "-           0);",
                "+           1);",
            ],
        )
        found = detect(parse_diff(diff), {})
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].trigger, TRIGGER_DEFAULT)
        self.assertEqual((found[0].old, found[0].new), ("0", "1"))
        self.assertEqual(found[0].line, 62)

    def test_reformatting_alone_is_not_a_change(self):
        diff = make_diff(
            "src/init.cc",
            "@@ -12,1 +12,2 @@",
            [
                '-NCCL_PARAM(Knob, "KNOB", 1);',
                '+NCCL_PARAM(Knob, "KNOB",',
                "+           1);",
            ],
        )
        self.assertEqual(detect(parse_diff(diff), {}), [])

    def test_declaration_moved_between_files_is_not_new(self):
        """A move shows as a removal in one file and an addition in another.

        The added side has no matching removal in its own hunk, so it reports as
        new. That is the intended conservative outcome: the reviewer confirms it
        with an entry or an exemption rather than the move passing silently.
        """
        diff = "\n".join(
            [
                make_diff(
                    "src/init.cc",
                    "@@ -12,1 +11,0 @@",
                    ['-NCCL_PARAM(Knob, "KNOB", 1);'],
                ),
                make_diff(
                    "src/enqueue.cc",
                    "@@ -5,0 +6,1 @@",
                    ['+NCCL_PARAM(Knob, "KNOB", 1);'],
                ),
            ]
        )
        found = detect(parse_diff(diff), {})
        self.assertEqual(
            [(d.env, d.trigger) for d in found], [("NCCL_KNOB", TRIGGER_NEW)]
        )


class ReferenceTest(unittest.TestCase):
    def test_accessor_resolves_to_the_env_var_name(self):
        diff = make_diff(
            "src/transport/p2p.cc",
            "@@ -80,0 +81,1 @@",
            ["+  if (ncclParamFoo()) return ncclSuccess;"],
        )
        found = detect(parse_diff(diff), registry_of(PARAM_FOO))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].env, "NCCL_FOO")
        self.assertEqual(found[0].trigger, TRIGGER_REFERENCE)
        self.assertEqual(found[0].line, 81)

    def test_unknown_accessor_is_reported_under_its_own_name(self):
        diff = make_diff(
            "src/init.cc", "@@ -1,0 +2,1 @@", ["+  int x = ncclParamMystery();"]
        )
        self.assertEqual(
            [d.env for d in detect(parse_diff(diff), {})], ["ncclParamMystery"]
        )

    def test_getenv_with_a_literal_name(self):
        diff = make_diff(
            "src/init.cc",
            "@@ -1,0 +2,1 @@",
            ['+  const char* v = getenv("NCCL_TOPO_FILE");'],
        )
        self.assertEqual(
            [d.env for d in detect(parse_diff(diff), {})], ["NCCL_TOPO_FILE"]
        )

    def test_getenv_with_a_computed_name_is_flagged_as_dynamic(self):
        diff = make_diff(
            "src/init.cc", "@@ -1,0 +2,1 @@", ["+  const char* v = ncclGetEnv(buf);"]
        )
        self.assertEqual([d.env for d in detect(parse_diff(diff), {})], ["<dynamic>"])

    def test_bare_env_literal(self):
        diff = make_diff(
            "src/graph/topo.cc",
            "@@ -1,0 +2,1 @@",
            ['+  setenvIfUnset("NCCL_MIN_NCHANNELS", "4");'],
        )
        self.assertEqual(
            [d.env for d in detect(parse_diff(diff), {})], ["NCCL_MIN_NCHANNELS"]
        )

    def test_comment_only_line_reads_nothing(self):
        diff = make_diff(
            "src/init.cc",
            "@@ -1,0 +2,1 @@",
            ["+  // ncclParamFoo() is consulted below"],
        )
        self.assertEqual(detect(parse_diff(diff), registry_of(PARAM_FOO)), [])

    def test_context_line_is_not_a_reference(self):
        diff = make_diff(
            "src/init.cc",
            "@@ -1,2 +1,3 @@",
            ["   if (ncclParamFoo()) {", "+    doSomethingElse();", "   }"],
        )
        self.assertEqual(detect(parse_diff(diff), registry_of(PARAM_FOO)), [])

    def test_declaration_outranks_its_own_reference(self):
        diff = make_diff(
            "src/init.cc",
            "@@ -1,0 +2,2 @@",
            ['+NCCL_PARAM(Foo, "FOO", 0);', "+  int v = ncclParamFoo();"],
        )
        found = detect(parse_diff(diff), registry_of(PARAM_FOO))
        self.assertEqual(
            [(d.env, d.trigger) for d in found], [("NCCL_FOO", TRIGGER_NEW)]
        )


class ScopeTest(unittest.TestCase):
    def test_test_directory_is_out_of_scope(self):
        diff = make_diff(
            "test/AllReduceTests.cpp",
            "@@ -1,0 +2,1 @@",
            ['+  setenv("NCCL_FOO", "1", 1);'],
        )
        self.assertEqual(detect(parse_diff(diff), {}), [])

    def test_tools_and_docs_are_out_of_scope(self):
        for path in ("tools/scripts/x.cc", "docs/x.h", "contrib/x.cc"):
            diff = make_diff(path, "@@ -1,0 +2,1 @@", ["+  int v = ncclParamFoo();"])
            self.assertEqual(detect(parse_diff(diff), registry_of(PARAM_FOO)), [], path)

    def test_plugins_and_tuner_are_in_scope(self):
        for path in ("plugins/profiler/inspector/inspector.cc", "tuner/x.cc"):
            diff = make_diff(path, "@@ -1,0 +2,1 @@", ['+  getenv("RCCL_INSPECTOR");'])
            self.assertEqual(
                [d.env for d in detect(parse_diff(diff), {})], ["RCCL_INSPECTOR"], path
            )

    def test_non_source_file_is_ignored(self):
        diff = make_diff("src/CMakeLists.txt", "@@ -1,0 +2,1 @@", ["+  # NCCL_FOO"])
        self.assertEqual(detect(parse_diff(diff), {}), [])


class DiffPathTest(unittest.TestCase):
    def test_strip_prefix_selects_the_rccl_subtree(self):
        diff = "\n".join(
            [
                make_diff(
                    "projects/rccl/src/init.cc",
                    "@@ -1,0 +2,1 @@",
                    ["+  int v = ncclParamFoo();"],
                ),
                make_diff(
                    "projects/amdsmi/src/x.cc",
                    "@@ -1,0 +2,1 @@",
                    ["+  int v = ncclParamFoo();"],
                ),
            ]
        )
        parsed = parse_diff(diff, "projects/rccl/")
        self.assertEqual([fd.path for fd in parsed], ["src/init.cc"])

    def test_deleted_file_has_no_new_side(self):
        diff = "\n".join(
            [
                "diff --git a/src/old.cc b/src/old.cc",
                "--- a/src/old.cc",
                "+++ /dev/null",
                "@@ -1,1 +0,0 @@",
                '-NCCL_PARAM(Knob, "KNOB", 1);',
            ]
        )
        self.assertEqual(detect(parse_diff(diff), {}), [])


class MarkdownCoverageTest(unittest.TestCase):
    MD = """# RCCL default-behavior changes

## Format

### New environment variables

| Variable | Supported | Description | Accepted values | Default | Reason for Change |
|---|---|---|---|---|---|
| `NCCL_EXAMPLE_ONLY` | Supported | example row | `0`, `1` | `0` | must not count |

## Unreleased

### New environment variables

| Variable | Supported | Description | Accepted values | Default | Reason for Change |
|---|---|---|---|---|---|
| `NCCL_NEW_KNOB` | Supported | Selects the X path | `0` (off), `1` (on) | `0` | Gates the new path |
| `RCCL_DBG_KNOB` | Not supported | Dumps scheduling decisions | `0` (off), `1` (on) | `0` | Bring-up aid |
| `NCCL_BLANK` | Supported | _what it controls_ | `0` (off), `1` (on) | `0` | Pasted, not filled in |
| `NCCL_ODD` | maybe | Selects the Y path | `0`, `1` | `0` | Bad support value |
| `NCCL_ANNOUNCED_ONLY` | | | | `0` | Row exists but is empty |

### Changed defaults and other behavior changes

| Change | Reason for Change |
|---|---|
| `NCCL_KNOB` default changed `1` -> `2` | Doubles the chunk count |
| `NCCL_STALE` default changed `3` -> `4` | From an older release |
| `NCCL_LISTED` added — default `0` | Only in the changes table |

## RCCL 2.30.4

### New environment variables

| Variable | Supported | Description | Accepted values | Default | Reason for Change |
|---|---|---|---|---|---|
| `NCCL_OLD` | Supported | Shipped previously | `0`, `1` | `1` | Historical |
"""

    def setUp(self):
        self.entries = parse_md_entries(self.MD)

    def gap(self, env, trigger, **kwargs):
        return coverage_gap(
            Detection(env, trigger, "src/x.cc", 1, "", **kwargs), self.entries
        )

    def test_example_rows_outside_a_release_section_do_not_count(self):
        names = {name for entry in self.entries for name in entry.names}
        self.assertNotIn("NCCL_EXAMPLE_ONLY", names)
        self.assertIn("NCCL_NEW_KNOB", names)
        self.assertIn("NCCL_OLD", names)

    def test_subsection_headings_do_not_close_the_release_section(self):
        names = {name for entry in self.entries for name in entry.names}
        self.assertIn("NCCL_KNOB", names)  # under the second ### of Unreleased

    def test_fully_documented_new_variable_passes(self):
        self.assertIsNone(self.gap("NCCL_NEW_KNOB", TRIGGER_NEW, new="0"))
        self.assertIsNone(self.gap("RCCL_DBG_KNOB", TRIGGER_NEW, new="0"))

    def test_undocumented_variable_is_not_covered(self):
        self.assertEqual(self.gap("NCCL_MISSING", TRIGGER_NEW, new="0"), NO_ENTRY)

    def test_new_variable_needs_the_new_variables_table(self):
        gap = self.gap("NCCL_LISTED", TRIGGER_NEW, new="0")
        self.assertIn("New environment variables table", gap)

    def test_placeholder_cells_do_not_count_as_documentation(self):
        gap = self.gap("NCCL_BLANK", TRIGGER_NEW, new="0")
        self.assertIn("Description", gap)

    def test_empty_cells_are_reported_by_name(self):
        gap = self.gap("NCCL_ANNOUNCED_ONLY", TRIGGER_NEW, new="0")
        for label in ("Supported", "Description", "Accepted values"):
            self.assertIn(label, gap)

    def test_support_status_must_use_the_agreed_vocabulary(self):
        gap = self.gap("NCCL_ODD", TRIGGER_NEW, new="0")
        self.assertIn("Supported is `maybe`", gap)

    def test_default_change_needs_the_new_value_in_the_row(self):
        self.assertIsNone(self.gap("NCCL_KNOB", TRIGGER_DEFAULT, old="1", new="2"))
        # Same variable, but the row documents an older 3 -> 4 change.
        gap = self.gap("NCCL_STALE", TRIGGER_DEFAULT, old="4", new="5")
        self.assertIn("does not mention the new default `5`", gap)

    def test_expression_default_falls_back_to_a_name_match(self):
        self.assertIsNone(
            self.gap("NCCL_STALE", TRIGGER_DEFAULT, old="1LL << 29", new="1LL << 30")
        )

    def test_reference_matches_by_name_without_the_full_schema(self):
        self.assertIsNone(self.gap("NCCL_LISTED", TRIGGER_REFERENCE))
        self.assertIsNone(self.gap("NCCL_OLD", TRIGGER_REFERENCE))

    def test_empty_file_covers_nothing(self):
        self.assertEqual(parse_md_entries(""), [])

    def test_shipped_file_parses_and_has_no_stale_entries(self):
        """The checked-in file must parse, and its examples must not count."""
        path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "rcclDefaultBehaviorChanges.md",
        )
        with open(path, encoding="utf-8") as handle:
            entries = parse_md_entries(handle.read())
        self.assertEqual(entries, [], "the Unreleased tables should start empty")


class ExemptionTest(unittest.TestCase):
    def test_em_dash_separator(self):
        exemptions, errors = parse_exemptions(
            "## Motivation\n\nBEHAVIOR-CHANGE-EXEMPT: NCCL_FOO, NCCL_BAR — refactor only\n"
        )
        self.assertEqual(errors, [])
        self.assertEqual(
            exemptions, {"NCCL_FOO": "refactor only", "NCCL_BAR": "refactor only"}
        )

    def test_hyphen_and_colon_separators(self):
        for body in (
            "BEHAVIOR-CHANGE-EXEMPT: NCCL_FOO - moved the call site",
            "BEHAVIOR-CHANGE-EXEMPT: NCCL_FOO: moved the call site",
        ):
            exemptions, errors = parse_exemptions(body)
            self.assertEqual(errors, [], body)
            self.assertEqual(exemptions, {"NCCL_FOO": "moved the call site"}, body)

    def test_missing_reason_is_an_error_not_a_pass(self):
        exemptions, errors = parse_exemptions("BEHAVIOR-CHANGE-EXEMPT: NCCL_FOO")
        self.assertEqual(exemptions, {})
        self.assertEqual(len(errors), 1)
        self.assertIn("NCCL_FOO", errors[0])

    def test_no_token_is_not_an_error(self):
        self.assertEqual(parse_exemptions("An ordinary PR description."), ({}, []))

    def test_quoted_line_still_counts(self):
        exemptions, _ = parse_exemptions("> BEHAVIOR-CHANGE-EXEMPT: NCCL_FOO — reason")
        self.assertEqual(exemptions, {"NCCL_FOO": "reason"})


class ReportTest(unittest.TestCase):
    def test_new_variable_row_has_the_full_documentation_schema(self):
        row = suggested_row(
            Detection("NCCL_FOO", TRIGGER_NEW, "src/x.cc", 1, "", new="0")
        )
        self.assertEqual(row.count("|"), 7)  # six columns
        self.assertIn("`NCCL_FOO`", row)
        self.assertIn("Supported", row)
        self.assertIn("_what it controls_", row)
        self.assertIn("`0`", row)  # the detected default, pre-filled

    def test_suggested_rows_name_the_variable_and_values(self):
        self.assertIn(
            "`NCCL_BAR` default changed `1` -> `2`",
            suggested_row(
                Detection(
                    "NCCL_BAR", TRIGGER_DEFAULT, "src/x.cc", 1, "", old="1", new="2"
                )
            ),
        )
        self.assertIn(
            "`NCCL_BAZ` newly referenced in `src/x.cc`",
            suggested_row(Detection("NCCL_BAZ", TRIGGER_REFERENCE, "src/x.cc", 1, "")),
        )

    def test_failure_report_carries_every_required_field(self):
        missing = [
            Detection(
                "NCCL_BAR", TRIGGER_DEFAULT, "src/init.cc", 12, "`1` -> `2`", "1", "2"
            ),
            Detection(
                "NCCL_BAZ", TRIGGER_REFERENCE, "src/p2p.cc", 81, "`ncclParamBaz`"
            ),
        ]
        report = build_report(missing, {}, [], "rcclDefaultBehaviorChanges.md")
        self.assertIn("NCCL_BAR", report)  # variable name
        self.assertIn("src/init.cc:12", report)  # file and line
        self.assertIn(TRIGGER_DEFAULT, report)  # which condition fired
        self.assertIn("`1` -> `2`", report)  # old and new default
        self.assertIn("| `NCCL_BAR` default changed `1` -> `2` | _why_ |", report)
        # Reference-only detections come with a ready-to-paste exemption line.
        self.assertIn("BEHAVIOR-CHANGE-EXEMPT: NCCL_BAZ", report)

    def test_new_variables_get_their_own_paste_block(self):
        missing = [
            Detection(
                "NCCL_NEW", TRIGGER_NEW, "src/init.cc", 12, "default `0`", new="0"
            ),
            Detection(
                "NCCL_BAZ", TRIGGER_REFERENCE, "src/p2p.cc", 81, "`ncclParamBaz`"
            ),
        ]
        report = build_report(
            missing,
            {},
            [],
            "rcclDefaultBehaviorChanges.md",
            {"NCCL_NEW": "the entry leaves Description empty"},
        )
        self.assertIn("### New environment variables", report)
        self.assertIn("| Variable | Supported | Description |", report)
        # The gap is spelled out, not just "add an entry".
        self.assertIn("the entry leaves Description empty", report)
        # The reference detection still routes to the changes table.
        self.assertIn("Changed defaults and other behavior changes", report)

    def test_applied_override_is_recorded_and_flagged(self):
        report = build_report(
            [],
            {
                "NCCL_FOO": (TRIGGER_REFERENCE, "moved"),
                "NCCL_BAR": (TRIGGER_NEW, "n/a"),
            },
            [],
            "rcclDefaultBehaviorChanges.md",
        )
        self.assertIn("Overrides applied", report)
        self.assertIn("| `NCCL_FOO` | reference | moved |", report)
        self.assertIn("**(this is a behavior change)**", report)


if __name__ == "__main__":
    unittest.main()
