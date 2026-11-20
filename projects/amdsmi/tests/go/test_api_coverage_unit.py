# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

if importlib.util.find_spec("test_api_coverage"):
    import test_api_coverage as coverage
else:
    coverage = None


def inventory() -> dict:
    return {
        "apis": {
            "amdsmi_read": {
                "signature": "amdsmi_status_t(uint32_t*)",
                "status": "deferred",
                "reason": "query",
            }
        },
        "reasons": {"query": "Awaiting the consumer API list and scope review."},
        "types": {},
    }


HEADER = "amdsmi_status_t amdsmi_read(uint32_t *value);"


class DeclarationTests(unittest.TestCase):
    def test_new_function_requires_inventory_entry(self) -> None:
        self.assertIsNotNone(coverage, "coverage comparator is missing")
        coverage.check_coverage(header=HEADER, source="", manifest=inventory())
        with self.assertRaisesRegex(ValueError, "new.*amdsmi_added"):
            coverage.check_coverage(
                header=HEADER + "\nvoid amdsmi_added(void);", source="", manifest=inventory()
            )

    def test_deleted_or_changed_signature_fails(self) -> None:
        for header, message in [
            ("", "removed.*amdsmi_read"),
            (HEADER.replace("amdsmi_status_t", "void"), "changed.*amdsmi_read"),
            (HEADER.replace("uint32_t", "uint64_t"), "changed.*amdsmi_read"),
            (HEADER.replace("uint32_t", "const uint32_t"), "changed.*amdsmi_read"),
            (HEADER.replace("*value", "**value"), "changed.*amdsmi_read"),
        ]:
            with self.subTest(header=header), self.assertRaisesRegex(ValueError, message):
                coverage.check_coverage(header=header, source="", manifest=inventory())

    def test_parameter_names_and_spacing_are_not_types(self) -> None:
        coverage.check_coverage(
            header="amdsmi_status_t\namdsmi_read( uint32_t* renamed );",
            source="",
            manifest=inventory(),
        )

    def test_unparsed_or_duplicate_declarations_fail_loud(self) -> None:
        for text in [
            HEADER + HEADER,
            "void amdsmi_callback(void (*callback)(int));",
            "void amdsmi_missing_semicolon(void)",
            "void amdsmi_old_style();",
            "amdsmi_status_t amdsmi_read(uint32_t *value) BROKEN;",
            "AMDSMI_DECLARE(amdsmi_hidden);",
            "typedef struct { int value; broken;",
        ]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "duplicate|unparsed"):
                coverage.declarations(text)

    def test_cpu_conditionals_and_nonstatus_returns_are_included(self) -> None:
        header = """
        #ifndef HEADER_GUARD
        #define HEADER_GUARD
        #ifdef __cplusplus
        extern "C" {
        #endif
        /* void amdsmi_comment(void); */
        typedef struct { int value; } amdsmi_info_t;
        #ifdef ENABLE_ESMI_LIB
        const char *amdsmi_cpu_name(unsigned int cpu, char name[16]);
        #endif
        void amdsmi_finish(void);
        #ifdef __cplusplus
        }
        #endif
        #endif
        """
        self.assertEqual(
            coverage.declarations(header),
            {"amdsmi_cpu_name": "const char*(unsigned int,char[16])", "amdsmi_finish": "void()"},
        )


class MappingTests(unittest.TestCase):
    def test_mixed_case_native_calls_require_bound_entries(self) -> None:
        name = "amdsmi_is_P2P_accessible"
        header = "void " + name + "(void);"
        for source in (
            "func Accessible() { C." + name + "() }",
            "/* static void go_amdsmi_p2p(void) { " + name + "(); } */\n"
            'import "C"\nfunc Accessible() { C.go_amdsmi_p2p() }',
        ):
            manifest = inventory()
            manifest["apis"] = {
                name: {"signature": "void()", "status": "deferred", "reason": "query"}
            }
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "native calls"):
                    coverage.check_coverage(header=header, source=source, manifest=manifest)
                with self.assertRaisesRegex(ValueError, "prohibited native calls"):
                    coverage.contract.check_native_calls(
                        coverage.contract.native_calls(source), available_only=True
                    )
                manifest["apis"][name].update(status="bound", go="Accessible")
                coverage.check_coverage(header=header, source=source, manifest=manifest)

    def test_c_preamble_helper_mapping(self) -> None:
        manifest = inventory()
        manifest["apis"]["amdsmi_read"].update(status="bound", go="Read", via="go_amdsmi_read")
        source = """
        /* static void go_amdsmi_read(void) { amdsmi_read(0); } */
        import "C"
        func Read() { C.go_amdsmi_read() }
        """
        coverage.check_coverage(header=HEADER, source=source, manifest=manifest)
        with self.assertRaisesRegex(ValueError, "mapping"):
            coverage.check_coverage(
                header=HEADER, source=source.replace("C.go_amdsmi_read()", ""), manifest=manifest
            )

    def test_bound_mapping_must_reach_actual_native_call(self) -> None:
        manifest = inventory()
        manifest["apis"]["amdsmi_read"].update(status="bound", go="Read", via="readLocked")
        source = """
        package amdsmi
        func Read() { readLocked() }
        func readLocked() { C.amdsmi_read(nil) }
        func Unrelated() {}
        """
        coverage.check_coverage(header=HEADER, source=source, manifest=manifest)
        for changed in [
            source.replace("func Read()", "func Renamed()"),
            source.replace("readLocked() }", "/* readLocked() */ }", 1),
            source.replace("C.amdsmi_read(nil)", '"C.amdsmi_read(nil)"'),
            source.replace("func Read()", "func notExported()") + "\n// func Read() {}",
            source.replace("func Read()", "func notExported()") + "\nvar s = `func Read() {}`",
        ]:
            with self.subTest(source=changed), self.assertRaisesRegex(ValueError, "mapping|calls"):
                coverage.check_coverage(header=HEADER, source=changed, manifest=manifest)
        manifest["apis"]["amdsmi_read"]["go"] = "Unrelated"
        with self.assertRaisesRegex(ValueError, "mapping"):
            coverage.check_coverage(header=HEADER, source=source, manifest=manifest)

    def test_invalid_status_or_missing_reason_is_not_accounted(self) -> None:
        for update in [
            {"status": "reviewed"},
            {"status": "backend-specific"},
            {"status": ""},
            {"reason": ""},
            {"reason": "unknown-category"},
            {"go": "Read"},
        ]:
            manifest = inventory()
            manifest["apis"]["amdsmi_read"].update(update)
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, "inventory"):
                coverage.check_coverage(header=HEADER, source="", manifest=manifest)
        manifest = inventory()
        manifest["reasons"]["query"] = "  "
        with self.assertRaisesRegex(ValueError, "reason"):
            coverage.check_coverage(header=HEADER, source="", manifest=manifest)

    def test_duplicate_json_rows_are_rejected(self) -> None:
        self.assertTrue(
            callable(getattr(coverage, "unique_object", None)), "duplicate-key check missing"
        )
        with self.assertRaisesRegex(ValueError, "duplicate.*amdsmi_read"):
            json.loads(
                '{"amdsmi_read": {}, "amdsmi_read": {}}', object_pairs_hook=coverage.unique_object
            )


class TypeTests(unittest.TestCase):
    def test_unbound_type_edits_do_not_change_layout_context(self) -> None:
        header = """
        #ifndef HEADER
        #define HEADER
        typedef struct { int value; } unused_t;
        typedef struct { int value; } amdsmi_info_t;
        void amdsmi_read(void);
        #endif
        """
        source = "func Read() { var out C.amdsmi_info_t; C.amdsmi_read() }"
        self.assertEqual(
            coverage.type_fingerprints(header, source),
            coverage.type_fingerprints(
                header.replace("int value; } unused_t", "int value; int added; } unused_t"), source
            ),
        )

    def test_packing_and_conditional_layout_changes_fail(self) -> None:
        header = "typedef struct { char name; int value; } amdsmi_info_t; void amdsmi_read(void);"
        source = "func Read() { var out C.amdsmi_info_t; C.amdsmi_read() }"
        manifest = inventory()
        manifest["apis"]["amdsmi_read"].update(signature="void()", status="bound", go="Read")
        manifest["types"] = coverage.type_fingerprints(header, source)
        for changed in (
            "#pragma pack(push, 1)\n" + header + "\n#pragma pack(pop)\n",
            header.replace("int value;", "\n#if 0\nint value;\n#endif\n"),
        ):
            with self.subTest(header=changed), self.assertRaisesRegex(ValueError, "types changed"):
                coverage.check_coverage(header=changed, source=source, manifest=manifest)

    def test_c_preamble_only_type_and_macro_drift(self) -> None:
        header = """
        #define AMDSMI_MIN_MAJOR 27
        typedef struct { int value; } amdsmi_helper_info_t;
        void amdsmi_read(void);
        """
        source = """
        /*
        static void go_amdsmi_read(void) {
            amdsmi_helper_info_t info = {AMDSMI_MIN_MAJOR};
            amdsmi_read();
        }
        */
        import "C"
        func Read() { C.go_amdsmi_read() }
        """
        manifest = inventory()
        manifest["apis"]["amdsmi_read"].update(
            signature="void()", status="bound", go="Read", via="go_amdsmi_read"
        )
        manifest["types"] = coverage.type_fingerprints(header, source)
        coverage.check_coverage(header=header, source=source, manifest=manifest)
        for changed in [
            header.replace("int value", "long value"),
            header.replace("AMDSMI_MIN_MAJOR 27", "AMDSMI_MIN_MAJOR 28"),
        ]:
            with self.subTest(header=changed), self.assertRaisesRegex(ValueError, "type.*changed"):
                coverage.check_coverage(header=changed, source=source, manifest=manifest)

    def test_typedef_aliases_do_not_require_amdsmi_prefix(self) -> None:
        header = (
            "typedef struct { int value; } native_info_t; typedef native_info_t processor_type_t;"
        )
        self.assertEqual(
            set(coverage.type_fingerprints(header, "var x C.processor_type_t")),
            {"native_info_t", "processor_type_t"},
        )

    def test_transitive_enum_struct_and_array_macro_drift(self) -> None:
        header = """
        #define AMDSMI_SIZE 2
        typedef enum { AMDSMI_A = 1, AMDSMI_B = 2 } amdsmi_kind_t;
        typedef struct { amdsmi_kind_t kind; char name[AMDSMI_SIZE]; } amdsmi_info_t;
        void amdsmi_info(amdsmi_info_t *info);
        """
        source = "func Info() { C.amdsmi_info(nil) }"
        self.assertTrue(
            callable(getattr(coverage, "type_fingerprints", None)), "type guard missing"
        )
        fingerprints = coverage.type_fingerprints(header, source)
        self.assertEqual(set(fingerprints), {"AMDSMI_SIZE", "amdsmi_kind_t", "amdsmi_info_t"})
        manifest = {
            "reasons": {"foundation": "Read-only foundation; consumer readiness is separate."},
            "types": fingerprints,
            "apis": {
                "amdsmi_info": {
                    "signature": "void(amdsmi_info_t*)",
                    "status": "bound",
                    "go": "Info",
                    "reason": "foundation",
                }
            },
        }
        coverage.check_coverage(header=header, source=source, manifest=manifest)
        for changed in [
            header.replace("AMDSMI_B = 2", "AMDSMI_B = 3"),
            header.replace("char name", "int name"),
            header.replace("#define AMDSMI_SIZE 2", "#define AMDSMI_SIZE 3"),
            header.replace("amdsmi_kind_t kind;", "amdsmi_kind_t kind; int extra;"),
        ]:
            with self.subTest(header=changed), self.assertRaisesRegex(ValueError, "type.*changed"):
                coverage.check_coverage(header=changed, source=source, manifest=manifest)
        manifest["types"].pop("amdsmi_kind_t")
        with self.assertRaisesRegex(ValueError, "type.*new"):
            coverage.check_coverage(header=header, source=source, manifest=manifest)


class CommandTests(unittest.TestCase):
    def test_checked_in_inventory_has_positive_offline_report(self) -> None:
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).with_name("test_api_coverage.py"))],
            cwd="/",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r"public declarations: \d+; bound: \d+; deferred: \d+")
        self.assertIn("not consumer readiness", result.stdout)


if __name__ == "__main__":
    unittest.main()
