# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavior tests for the rocjitsu stylist command."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIRECTORY = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from stylist_rules import Violation, check_style_rules, normalize_direct_rules

STYLIST = SCRIPTS_DIRECTORY / "stylist.py"
STYLIST_MODULE = SourceFileLoader("stylist_module", str(STYLIST)).load_module()
StyleResult = STYLIST_MODULE.StyleResult
_new_violations = STYLIST_MODULE._new_violations
_formatted_contents = STYLIST_MODULE._formatted_contents


class StylistTest(unittest.TestCase):
    """Exercise the stylist command-line behavior against temporary files."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(self.temporary_directory.cleanup)
        self.test_directory = Path(self.temporary_directory.name)

    def run_stylist(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(STYLIST), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_check_reports_problem_without_changing_file(self) -> None:
        source_file = self.test_directory / "needs_formatting.cpp"
        original_contents = "int main(){return 0;}\n"
        source_file.write_text(original_contents)

        result = self.run_stylist("--check", str(source_file))

        self.assertEqual(result.returncode, 1)
        self.assertIn("needs formatting", result.stderr)
        self.assertEqual(source_file.read_text(), original_contents)

    def test_fix_is_idempotent_and_returns_success(self) -> None:
        source_file = self.test_directory / "fixable.cpp"
        source_file.write_text(
            "template <class ValueType> ValueType identity(ValueType value)"
            "{return value;}\n"
        )

        fix_result = self.run_stylist(str(source_file))
        contents_after_fix = source_file.read_bytes()
        second_fix_result = self.run_stylist(str(source_file))

        self.assertEqual(fix_result.returncode, 0)
        self.assertEqual(second_fix_result.returncode, 0)
        self.assertEqual(source_file.read_bytes(), contents_after_fix)
        self.assertIn("template <typename ValueType>", source_file.read_text())
        self.assertEqual(self.run_stylist("--check", str(source_file)).returncode, 0)

    def test_check_reports_template_class_without_changing_file(self) -> None:
        source_file = self.test_directory / "template_parameter.h"
        original_contents = "template <class ValueType> class Container {};\n"
        source_file.write_text(original_contents)

        result = self.run_stylist("--check", str(source_file))

        self.assertEqual(result.returncode, 1)
        self.assertEqual(source_file.read_text(), original_contents)

    def test_fix_preserves_class_text_inside_template_declaration(self) -> None:
        source_file = self.test_directory / "template_text.h"
        source_file.write_text(
            'template <class ValueType, /* class CommentType */ '
            'const char *Description = "class StringType"> class Container {};\n'
        )

        result = self.run_stylist(str(source_file))
        output = source_file.read_text()

        self.assertEqual(result.returncode, 0)
        self.assertIn("typename ValueType", output)
        self.assertIn("class CommentType", output)
        self.assertIn('"class StringType"', output)
        self.assertIn("class Container", output)

    def test_fix_handles_parameter_after_nested_template_default(self) -> None:
        source_file = self.test_directory / "nested_template.h"
        source_file.write_text(
            "template <class ValueType = Container<int>, class AllocatorType> "
            "class Wrapper {};\n"
        )

        result = self.run_stylist(str(source_file))
        output = source_file.read_text()

        self.assertEqual(result.returncode, 0)
        self.assertIn("typename ValueType = Container<int>", output)
        self.assertIn("typename AllocatorType", output)
        self.assertIn("class Wrapper", output)

    def test_fix_replaces_only_outer_header_guard(self) -> None:
        source_file = self.test_directory / "guarded.h"
        source_file.write_text(
            "#ifndef PROJECT_GUARDED_H_\n"
            "#define PROJECT_GUARDED_H_\n"
            "#ifdef FEATURE_ENABLED\n"
            "int feature();\n"
            "#endif\n"
            "#endif // PROJECT_GUARDED_H_\n"
        )

        result = self.run_stylist(str(source_file))
        output = source_file.read_text()

        self.assertEqual(result.returncode, 0)
        self.assertTrue(output.startswith("#pragma once\n"))
        self.assertIn("#ifdef FEATURE_ENABLED", output)
        self.assertEqual(output.count("#endif"), 1)
        self.assertEqual(self.run_stylist("--check", str(source_file)).returncode, 0)

    def test_directory_inputs_are_processed_in_parallel(self) -> None:
        first_source = self.test_directory / "first.cpp"
        second_source = self.test_directory / "nested" / "second.h"
        second_source.parent.mkdir()
        first_source.write_text("int first(){return 1;}\n")
        second_source.write_text("int second(){return 2;}\n")
        (self.test_directory / "ignored.txt").write_text("not C++\n")

        result = self.run_stylist("--jobs", "2", str(self.test_directory))

        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            self.run_stylist("--check", str(self.test_directory)).returncode,
            0,
        )

    def test_missing_path_is_an_operational_error(self) -> None:
        result = self.run_stylist(str(self.test_directory / "missing.cpp"))

        self.assertEqual(result.returncode, 2)
        self.assertIn("no such file or directory", result.stderr)

    def test_invalid_utf8_is_an_operational_error_in_both_modes(self) -> None:
        source_file = self.test_directory / "invalid_utf8.cpp"
        original_contents = b"int value = 0;\n// \xff\n"
        source_file.write_bytes(original_contents)

        check_result = self.run_stylist("--check", str(source_file))
        fix_result = self.run_stylist(str(source_file))

        self.assertEqual(check_result.returncode, 2)
        self.assertEqual(fix_result.returncode, 2)
        self.assertIn("utf-8", check_result.stderr)
        self.assertIn("utf-8", fix_result.stderr)
        self.assertEqual(source_file.read_bytes(), original_contents)

    def test_invalid_utf8_from_clang_format_is_an_operational_error(self) -> None:
        fake_clang_format = self.test_directory / "fake-clang-format"
        fake_clang_format.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "sys.stdout.buffer.write(b'\\xff')\n"
        )
        fake_clang_format.chmod(0o755)

        formatted_contents, error = _formatted_contents(
            str(fake_clang_format), self.test_directory / "source.cpp", b"int value;\n"
        )

        self.assertIsNone(formatted_contents)
        self.assertIsNotNone(error)
        self.assertIn("invalid UTF-8", error)

    def test_excessive_job_count_is_rejected(self) -> None:
        result = self.run_stylist("--jobs", "257", str(self.test_directory))

        self.assertEqual(result.returncode, 2)
        self.assertIn("--jobs must not exceed 256", result.stderr)

    def test_fix_removes_decorative_comments_and_uses_cpp_headers(self) -> None:
        source_file = self.test_directory / "direct_rules.cpp"
        source_file.write_text(
            "#include <stdio.h>\n" "\n" "// ----------------\n" "int value = 0;\n"
        )

        check_result = self.run_stylist("--check", str(source_file))
        fix_result = self.run_stylist(str(source_file))
        output = source_file.read_text()

        self.assertEqual(check_result.returncode, 1)
        self.assertEqual(fix_result.returncode, 0)
        self.assertIn("#include <cstdio>", output)
        self.assertNotIn("----------------", output)

    def test_struct_with_behavior_is_reported_but_pod_struct_is_allowed(self) -> None:
        source_file = self.test_directory / "structs.h"
        source_file.write_text(
            "struct Point {\n"
            "  int x;\n"
            "  int y;\n"
            "};\n"
            "\n"
            "struct Buffer {\n"
            "  Buffer();\n"
            "};\n"
        )

        result = self.run_stylist("--check", str(source_file))

        self.assertEqual(result.returncode, 1)
        self.assertIn("[struct-pod]", result.stderr)
        self.assertIn("struct Buffer", result.stderr)
        self.assertNotIn("struct Point", result.stderr)

    def test_naming_violations_are_reported(self) -> None:
        source_file = self.test_directory / "naming.h"
        source_file.write_text(
            "class bad_type {\n" "public:\n" "  void BadMethod();\n" "};\n"
        )

        result = self.run_stylist("--check", str(source_file))

        self.assertEqual(result.returncode, 1)
        self.assertIn("[type-name]", result.stderr)
        self.assertIn("[method-name]", result.stderr)

    def test_forbidden_logging_ignores_comments_and_literals(self) -> None:
        source_file = self.test_directory / "logging.cpp"
        source_file.write_text(
            "void log_message() {\n"
            "  // printf(\"comment\");\n"
            '  const char *message = "std::cerr";\n'
            '  std::printf("%s", message);\n'
            "}\n"
        )

        result = self.run_stylist("--check", str(source_file))

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr.count("[logging]"), 1)
        self.assertIn("std::printf", result.stderr)

    def test_strict_mode_requires_complete_public_api_doxygen(self) -> None:
        source_file = self.test_directory / "documented.h"
        source_file.write_text(
            "/// @brief A documented class.\n"
            "/// @details The class exists to test strict API documentation.\n"
            "class Documented {\n"
            "public:\n"
            "  void missing_documentation();\n"
            "};\n"
        )

        normal_result = self.run_stylist("--check", str(source_file))
        strict_result = self.run_stylist("--strict", "--check", str(source_file))

        self.assertEqual(normal_result.returncode, 0)
        self.assertEqual(strict_result.returncode, 1)
        self.assertIn("[doxygen]", strict_result.stderr)
        self.assertIn("@brief and @details", strict_result.stderr)

    def test_generated_files_skip_semantic_checks(self) -> None:
        source_file = self.test_directory / "generated.h"
        source_file.write_text(
            "// AUTO-GENERATED by the amdisa codegen pipeline. DO NOT EDIT.\n"
            "struct GeneratedBehavior {\n"
            "  GeneratedBehavior();\n"
            "};\n"
        )

        result = self.run_stylist("--strict", "--check", str(source_file))

        self.assertEqual(result.returncode, 0)

    def test_format_only_skips_semantic_violations(self) -> None:
        source_file = self.test_directory / "legacy.cpp"
        source_file.write_text('void run() { std::printf("legacy"); }\n')

        normal_result = self.run_stylist("--check", str(source_file))
        format_only_result = self.run_stylist(
            "--format-only", "--check", str(source_file)
        )

        self.assertEqual(normal_result.returncode, 1)
        self.assertIn("[logging]", normal_result.stderr)
        self.assertEqual(format_only_result.returncode, 0)

    def test_strict_and_format_only_are_incompatible(self) -> None:
        result = self.run_stylist("--strict", "--format-only")

        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot be used together", result.stderr)


class RuleEngineTest(unittest.TestCase):
    """Exercise individual semantic checks without invoking clang-format."""

    def test_namespace_must_match_owning_module(self) -> None:
        violations = check_style_rules(
            Path("/project/lib/util/include/util/wrong.h"),
            "namespace rocjitsu {\nint value;\n}\n",
        )

        self.assertEqual([violation.rule for violation in violations], ["namespace"])
        self.assertIn("must match module util", violations[0].message)

    def test_matching_module_namespace_is_allowed(self) -> None:
        violations = check_style_rules(
            Path("/project/lib/util/include/util/right.h"),
            "namespace util::detail {\nint value;\n}\n",
        )

        self.assertEqual(violations, [])

    def test_c_api_and_external_headers_are_not_rewritten(self) -> None:
        contents = "#include <stdint.h>\n"

        c_api_output = normalize_direct_rules(
            Path("/project/lib/rocjitsu/include/rocjitsu/api.h"), contents
        )
        external_output = normalize_direct_rules(
            Path("/project/lib/rocjitsu/external_headers/vendor.h"), contents
        )

        self.assertEqual(c_api_output, contents)
        self.assertEqual(external_output, contents)

    def test_member_suffix_and_class_section_rules(self) -> None:
        violations = check_style_rules(
            Path("/project/lib/util/include/util/example.h"),
            "class Example {\n"
            "private:\n"
            "  int missing_suffix;\n"
            "public:\n"
            "  int public_value_;\n"
            "private:\n"
            "  int repeated_;\n"
            "};\n",
        )

        rules = [violation.rule for violation in violations]
        self.assertIn("member-suffix", rules)
        self.assertIn("class-sections", rules)

    def test_macro_and_global_constant_naming(self) -> None:
        violations = check_style_rules(
            Path("/project/lib/util/src/example.cpp"),
            "#define LOCAL_MACRO 1\n"
            "constexpr int max_threads = 64;\n"
            "#define RJ_COMPAT_MACRO 2\n"
            "constexpr int kMaxThreads = 64;\n",
        )

        rules = [violation.rule for violation in violations]
        self.assertEqual(rules.count("macro"), 1)
        self.assertEqual(rules.count("constant-name"), 1)

    def test_counted_baseline_suppresses_only_existing_occurrences(self) -> None:
        path = Path("/project/example.cpp")
        violation = Violation(5, "logging", "printf is forbidden")
        key = f"{path}\tlogging\tprintf is forbidden"
        results = [StyleResult(path=path, violations=(violation, violation))]

        new_violations = _new_violations(results, {key: 1})

        self.assertEqual(new_violations, [(path, violation)])


if __name__ == "__main__":
    unittest.main()
