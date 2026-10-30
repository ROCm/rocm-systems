#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import abi_gate

try:
    import clang.cindex  # noqa: F401

    HAVE_LIBCLANG = True
except ImportError:
    HAVE_LIBCLANG = False

PRELUDE = "#include <stdint.h>\n#include <stdbool.h>\n#define AMDSMI_LIB_VERSION_MAJOR 27\n"
EXTERN_C = '#ifdef __cplusplus\nextern "C" {\n#endif\n'
END_EXTERN_C = "#ifdef __cplusplus\n}\n#endif\n"

BASE = (
    PRELUDE
    + EXTERN_C
    + """
#define AMDSMI_MAX_THING 16
#define AMDSMI_TIME_FORMAT "%02d"
typedef enum { AMDSMI_STATUS_SUCCESS = 0, AMDSMI_STATUS_INVAL = 1, AMDSMI_BLOCK_LAST = 2 } amdsmi_status_t;
typedef void* amdsmi_processor_handle;
typedef struct {
  uint32_t version;  //!< Output: layout version
  uint64_t value;
  uint32_t reserved[8];  //!< Reserved
} amdsmi_info_t;
/**
 *  @param[in] handle a handle
 *  @param[out] info filled on return
 */
amdsmi_status_t amdsmi_get_info(amdsmi_processor_handle handle, amdsmi_info_t* info);
amdsmi_status_t amdsmi_old_api(void);
"""
    + END_EXTERN_C
)


def edit(text, old, new):
    assert old in text, old
    return text.replace(old, new)


@unittest.skipUnless(HAVE_LIBCLANG, "libclang Python bindings not installed")
class DiffTest(unittest.TestCase):
    def breaks(self, new_text, old_text=BASE, policy=None):
        old = abi_gate.snapshot_text(old_text)
        new = abi_gate.snapshot_text(new_text)
        return abi_gate.diff(old, new, policy)

    def paths(self, new_text, **kw):
        return sorted(abi_gate.path_str(b.path) for b in self.breaks(new_text, **kw))

    def test_snapshot_records_layout(self):
        snap = abi_gate.snapshot_text(BASE)
        info = snap["types"]["amdsmi_info_t"]
        self.assertEqual((info["size"], info["align"]), (48, 8))
        self.assertEqual(info["fields"][1], ["value", 8, 8, "unsigned long"])
        self.assertEqual(
            snap["functions"]["amdsmi_get_info"], "amdsmi_status_t (void *, amdsmi_info_t *)"
        )
        self.assertEqual(snap["macros"]["AMDSMI_MAX_THING"], 16)
        self.assertEqual(snap["macros"]["AMDSMI_TIME_FORMAT"], {"text": '"%02d"'})

    def test_identical_header_passes(self):
        self.assertEqual(self.paths(BASE), [])

    def test_additions_pass(self):
        new = BASE + "#define AMDSMI_NEW 1\namdsmi_status_t amdsmi_new_api(void);\n"
        new = edit(
            new, "AMDSMI_STATUS_INVAL = 1,", "AMDSMI_STATUS_INVAL = 1, AMDSMI_STATUS_BUSY = 5,"
        )
        self.assertEqual(self.paths(new), [])

    def test_carving_reserved_passes(self):
        new = edit(BASE, "uint32_t reserved[8];", "uint32_t mask; uint32_t reserved[7];")
        self.assertEqual(self.paths(new), [])

    def test_growth_names_size_and_moved_reserved(self):
        new = edit(BASE, "uint64_t value;", "uint64_t value; uint64_t extra;")
        self.assertEqual(
            self.paths(new), ["types/amdsmi_info_t/fields/reserved", "types/amdsmi_info_t/size"]
        )
        rows = [r for b in self.breaks(new) for r in b.rows]
        self.assertIn(["reserved", "offset", "16", "24"], rows)
        self.assertIn(["", "size", "48", "56"], rows)

    def test_carving_a_reserved_bitfield_passes(self):
        old = PRELUDE + "typedef struct { uint32_t a : 4; uint32_t reserved : 28; } amdsmi_f_t;\n"
        new = edit(old, "uint32_t reserved : 28;", "uint32_t b : 1; uint32_t reserved : 27;")
        self.assertEqual(self.paths(new, old_text=old), [])
        grown = edit(old, "a : 4; uint32_t reserved : 28;", "a : 5; uint32_t reserved : 27;")
        self.assertEqual(self.paths(grown, old_text=old), ["types/amdsmi_f_t/fields/a"])

    def test_carving_a_bitfield_from_a_reserved_array_passes(self):
        new = edit(
            BASE, "uint32_t reserved[8];", "uint32_t f : 1; uint32_t r : 31; uint32_t reserved[7];"
        )
        self.assertEqual(self.paths(new), [])

    def test_carve_that_grows_struct_is_a_break(self):
        new = edit(BASE, "uint32_t reserved[8];", "uint32_t mask; uint32_t reserved[8];")
        self.assertIn("types/amdsmi_info_t/size", self.paths(new))

    def test_field_type_change_same_size(self):
        new = edit(BASE, "uint32_t version;", "int32_t version;")
        self.assertEqual(self.paths(new), ["types/amdsmi_info_t/fields/version"])

    def test_enumerator_value_change_and_removal(self):
        new = edit(BASE, "AMDSMI_STATUS_INVAL = 1,", "AMDSMI_STATUS_INVAL = 3,")
        self.assertEqual(self.paths(new), ["types/amdsmi_status_t/enumerators/AMDSMI_STATUS_INVAL"])
        new = edit(BASE, "AMDSMI_STATUS_INVAL = 1,", "")
        self.assertEqual(self.paths(new), ["types/amdsmi_status_t/enumerators/AMDSMI_STATUS_INVAL"])

    def test_end_sentinel_may_grow(self):
        new = edit(BASE, "AMDSMI_BLOCK_LAST = 2", "AMDSMI_X = 2, AMDSMI_BLOCK_LAST = 3")
        self.assertEqual(self.paths(new), [])
        notes = abi_gate.doc_changes(abi_gate.snapshot_text(BASE), abi_gate.snapshot_text(new))
        self.assertIn(["amdsmi_status_t.AMDSMI_BLOCK_LAST", "sentinel", 2, 3], notes)

    def test_function_removed_or_changed(self):
        new = edit(BASE, "amdsmi_status_t amdsmi_old_api(void);", "")
        self.assertEqual(self.paths(new), ["functions/amdsmi_old_api"])
        new = edit(BASE, "amdsmi_old_api(void)", "amdsmi_old_api(int flags)")
        self.assertEqual(self.paths(new), ["functions/amdsmi_old_api"])

    def test_static_inline_replacing_a_declaration_blocks(self):
        new = edit(
            BASE,
            "amdsmi_status_t amdsmi_old_api(void);",
            "static inline amdsmi_status_t amdsmi_old_api(void) { return AMDSMI_STATUS_SUCCESS; }",
        )
        self.assertEqual(self.paths(new), ["functions/amdsmi_old_api"])

    def test_typedef_rename_of_same_type_passes(self):
        new = edit(BASE, "uint32_t version;", "unsigned int version;")
        self.assertEqual(self.paths(new), [])

    def test_macro_change_blocks_but_version_does_not(self):
        new = edit(BASE, "AMDSMI_MAX_THING 16", "AMDSMI_MAX_THING 32")
        self.assertEqual(self.paths(new), ["macros/AMDSMI_MAX_THING"])
        new = edit(BASE, "AMDSMI_LIB_VERSION_MAJOR 27", "AMDSMI_LIB_VERSION_MAJOR 28")
        self.assertEqual(self.paths(new), [])
        new = edit(BASE, '"%02d"', '"%03d"')
        self.assertEqual(self.paths(new), [])

    def test_append_only_policy(self):
        new = edit(BASE, "uint32_t reserved[8];", "uint32_t reserved[8]; uint64_t tail;")
        self.assertEqual(
            self.paths(new, policy={"append_only": {"amdsmi_info_t": "size header"}}), []
        )
        self.assertEqual(self.paths(new), ["types/amdsmi_info_t/size"])

    def test_nested_anonymous_records_are_named(self):
        text = (
            PRELUDE
            + "typedef union { struct { uint32_t a : 3; uint32_t b : 5; }; uint32_t raw; } amdsmi_bits_t;\n"
        )
        text += "typedef struct { struct { uint32_t x; } inner; } amdsmi_outer_t;\n"
        snap = abi_gate.snapshot_text(text)
        self.assertEqual(
            snap["types"]["struct amdsmi_bits_t::(anon #0)"]["fields"][1],
            ["b", "0:3", "5b", "unsigned int"],
        )
        self.assertEqual(
            snap["types"]["amdsmi_outer_t"]["fields"][0][3], "struct amdsmi_outer_t::inner"
        )
        self.assertIn("struct amdsmi_outer_t::inner", snap["types"])

    def test_unknown_include_is_refused(self):
        with self.assertRaises(abi_gate.GateError):
            abi_gate.snapshot_text('#include "/etc/passwd"\n')

    def test_parse_error_is_refused(self):
        with self.assertRaises(abi_gate.GateError):
            abi_gate.snapshot_text("typedef struct { undefined_t x; } amdsmi_bad_t;\n")

    def test_quoted_include_of_a_shim_name_is_refused(self):
        with self.assertRaises(abi_gate.GateError):
            abi_gate.snapshot_text(BASE + '#include "stdint.h"\n')

    def test_unknown_conditional_is_refused_but_include_guard_is_not(self):
        with self.assertRaisesRegex(abi_gate.GateError, "_WIN32"):
            abi_gate.snapshot_text(BASE + "#ifdef _WIN32\ntypedef int amdsmi_w_t;\n#endif\n")
        guarded = "#ifndef MY_GUARD_H_\n#define MY_GUARD_H_\n" + BASE + "#endif\n"
        self.assertIn("amdsmi_info_t", abi_gate.snapshot_text(guarded)["types"])

    def test_attribute_before_brace_still_records_fields(self):
        text = (
            PRELUDE
            + "typedef struct __attribute__((__packed__)) { uint8_t a; uint32_t b; } amdsmi_p_t;\n"
        )
        snap = abi_gate.snapshot_text(text)
        self.assertEqual(snap["types"]["amdsmi_p_t"]["fields"][1], ["b", 1, 4, "unsigned int"])

    def test_renamed_typedef_kept_as_alias_passes(self):
        old = PRELUDE + "typedef enum { A0 = 0 } old_t;\nint amdsmi_f(old_t* p);\n"
        new = (
            PRELUDE
            + "typedef enum { A0 = 0 } amdsmi_new_t;\ntypedef amdsmi_new_t old_t;\nint amdsmi_f(amdsmi_new_t* p);\n"
        )
        self.assertEqual(self.paths(new, old_text=old), [])
        self.assertEqual(abi_gate.snapshot_text(new)["aliases"], {"old_t": "amdsmi_new_t"})

    def test_macro_that_becomes_an_enumerator_is_advisory(self):
        new = edit(BASE, "#define AMDSMI_MAX_THING 16", "enum { AMDSMI_MAX_THING = 16 };")
        self.assertEqual(self.paths(new), [])
        notes = abi_gate.doc_changes(abi_gate.snapshot_text(BASE), abi_gate.snapshot_text(new))
        self.assertEqual(notes[0][:2], ["AMDSMI_MAX_THING", "macro became enumerator"])

    def test_function_like_macro_change_blocks(self):
        old = BASE + "#define AMDSMI_MASK(i) (1ULL << ((i) - 1))\n"
        new = BASE + "#define AMDSMI_MASK(i) (1ULL << (i))\n"
        self.assertEqual(self.paths(new, old_text=old), ["macros/AMDSMI_MASK"])
        self.assertEqual(
            self.paths(BASE + "#define AMDSMI_MASK(i)  (1ULL<<((i)-1))\n", old_text=old), []
        )

    def test_inline_body_change_is_advisory(self):
        old = BASE + "static inline int amdsmi_k(int i) { return i; }\n"
        new = BASE + "static inline int amdsmi_k(int i) { return i + 1; }\n"
        self.assertEqual(self.paths(new, old_text=old), [])
        notes = abi_gate.doc_changes(abi_gate.snapshot_text(old), abi_gate.snapshot_text(new))
        self.assertEqual([n[:2] for n in notes], [["amdsmi_k", "inline body"]])


@unittest.skipUnless(HAVE_LIBCLANG, "libclang Python bindings not installed")
class EnumRuleTest(unittest.TestCase):
    ENUM = PRELUDE + "typedef enum { E_A = 0, E_B = 1, E_C = 4, E_LAST = 4 } amdsmi_e_t;\n"

    def check(self, new, retired=None):
        old = abi_gate.snapshot_text(self.ENUM)
        new = abi_gate.snapshot_text(new)
        policy = {"retired": retired or {}}
        return sorted(abi_gate.path_str(b.path) for b in abi_gate.diff(old, new, policy)), [
            n[:2] for n in abi_gate.doc_changes(old, new)
        ]

    def test_appending_passes_and_moves_the_sentinel(self):
        paths, notes = self.check(edit(self.ENUM, "E_LAST = 4", "E_D = 5, E_LAST = 5"))
        self.assertEqual(paths, [])
        self.assertEqual(notes, [["amdsmi_e_t.E_LAST", "sentinel"]])

    def test_reusing_a_removed_value_blocks(self):
        paths, _ = self.check(edit(self.ENUM, "E_B = 1", "E_X = 1"))
        self.assertEqual(
            paths, ["types/amdsmi_e_t/enumerators/E_B", "types/amdsmi_e_t/enumerators/E_X"]
        )

    def test_reusing_a_retired_value_blocks(self):
        paths, _ = self.check(edit(self.ENUM, "E_C = 4,", "E_C = 4, E_Y = 2,"), {"amdsmi_e_t": {2}})
        self.assertEqual(paths, ["types/amdsmi_e_t/enumerators/E_Y"])

    def test_reusing_a_value_renumbered_away_blocks(self):
        paths, _ = self.check(edit(self.ENUM, "E_A = 0,", "E_A = 3, E_N = 0,"))
        self.assertEqual(
            paths, ["types/amdsmi_e_t/enumerators/E_A", "types/amdsmi_e_t/enumerators/E_N"]
        )

    def test_aliased_and_count_end_markers_may_grow(self):
        for old, new in (
            ("X_A = 0, X_B = 1, X_LAST = X_B", "X_A = 0, X_B = 1, X_C = 2, X_LAST = X_C"),
            ("X_A, X_B, X_COUNT", "X_A, X_B, X_C, X_COUNT"),
        ):
            enum = PRELUDE + "typedef enum { %s } amdsmi_x_t;\n"
            paths = abi_gate.diff(
                abi_gate.snapshot_text(enum % old), abi_gate.snapshot_text(enum % new)
            )
            self.assertEqual(paths, [], old)

    def test_an_inner_range_bound_may_not_grow(self):
        old = edit(self.ENUM, "E_B = 1,", "E_B = 1, E_B_LAST = 1,")
        new = edit(self.ENUM, "E_B = 1,", "E_B = 1, E_B2 = 2, E_B_LAST = 2,")
        self.assertEqual(
            [
                abi_gate.path_str(b.path)
                for b in abi_gate.diff(abi_gate.snapshot_text(old), abi_gate.snapshot_text(new))
            ],
            ["types/amdsmi_e_t/enumerators/E_B_LAST"],
        )

    def test_only_the_highest_end_marker_may_grow(self):
        real = (
            PRELUDE + "typedef enum { F_MIN = 0, F_MAX = 1, F_INVALID = 0xFFFFFFFF } amdsmi_f_t;\n"
        )
        old, new = real, edit(real, "F_MAX = 1", "F_MAX = 2")
        self.assertEqual(
            abi_gate.diff(abi_gate.snapshot_text(old), abi_gate.snapshot_text(new))[0].path,
            ("types", "amdsmi_f_t", "enumerators", "F_MAX"),
        )
        self.assertIn("F_MAX == 1ULL", abi_gate.render_asserts(abi_gate.snapshot_text(old)))
        old = edit(self.ENUM, "E_B = 1,", "E_B = 1, E_B_END = 1,")
        new = edit(self.ENUM, "E_B = 1,", "E_B = 1, E_B2 = 2, E_B_END = 2,")
        diff = abi_gate.diff(abi_gate.snapshot_text(old), abi_gate.snapshot_text(new))
        self.assertEqual(
            [b.path for b in diff], [("types", "amdsmi_e_t", "enumerators", "E_B_END")]
        )
        self.assertIn("E_LAST >= 4ULL", abi_gate.render_asserts(abi_gate.snapshot_text(old)))

    def test_insert_into_a_gap_alias_and_overflow_are_advisory(self):
        new = edit(self.ENUM, "E_C = 4,", "E_C = 4, E_G = 2, E_ALIAS = 1, E_OVER = 9,")
        paths, notes = self.check(new)
        self.assertEqual(paths, [])
        self.assertIn(["amdsmi_e_t.E_G", "not appended"], notes)
        self.assertIn(["amdsmi_e_t.E_ALIAS", "alias"], notes)
        self.assertIn(["amdsmi_e_t.E_OVER", "beyond sentinel"], notes)


@unittest.skipUnless(HAVE_LIBCLANG, "libclang Python bindings not installed")
class LayoutHashTest(unittest.TestCase):
    def hashes(self, text):
        return abi_gate.snapshot_text(text)["types"]

    def test_same_size_reorder_changes_the_hash(self):
        old = self.hashes(BASE)["amdsmi_info_t"]["layout_hash"]
        new = self.hashes(edit(BASE, "uint32_t version;", "uint32_t version2;"))
        self.assertNotEqual(old, new["amdsmi_info_t"]["layout_hash"])
        self.assertEqual(old, self.hashes(BASE)["amdsmi_info_t"]["layout_hash"])

    def test_nested_record_change_reaches_the_outer_hash(self):
        text = PRELUDE + "typedef struct { uint32_t a; uint32_t b; } amdsmi_in_t;\n"
        text += "typedef struct { amdsmi_in_t in[2]; } amdsmi_out_t;\n"
        old = self.hashes(text)["amdsmi_out_t"]["layout_hash"]
        new = self.hashes(edit(text, "uint32_t a; uint32_t b;", "uint32_t b; uint32_t a;"))
        self.assertNotEqual(old, new["amdsmi_out_t"]["layout_hash"])

    def test_hash_recomputes_from_the_recorded_fields(self):
        types = self.hashes(BASE)
        recomputed = abi_gate.layout_hashes(types)
        self.assertEqual(recomputed["amdsmi_info_t"], types["amdsmi_info_t"]["layout_hash"])
        self.assertEqual(recomputed["amdsmi_status_t"], types["amdsmi_status_t"]["layout_hash"])


SEM = (
    PRELUDE
    + """
typedef enum { AMDSMI_STATUS_SUCCESS = 0 } amdsmi_status_t;
typedef void* amdsmi_processor_handle;
typedef struct {
  uint32_t power;  //!< Socket power in W
  uint32_t flags;  //!< Feature bitmask
  uint32_t clk;    //!< Clock in MHz
} amdsmi_pwr_t;
/**
 *  @param[in] handle a handle
 *  @param[out] info filled in, power in watts
 *  @param[out] limit limit in W
 */
amdsmi_status_t amdsmi_get_pwr(amdsmi_processor_handle handle, amdsmi_pwr_t* info, uint32_t* limit);
"""
)


@unittest.skipUnless(HAVE_LIBCLANG, "libclang Python bindings not installed")
class SemanticTest(unittest.TestCase):
    def paths(self, new, old=SEM):
        return sorted(
            abi_gate.path_str(b.path)
            for b in abi_gate.diff(abi_gate.snapshot_text(old), abi_gate.snapshot_text(new))
        )

    def test_annotations_are_seeded_from_docs(self):
        sem = abi_gate.snapshot_text(SEM)["semantics"]
        self.assertEqual(sem["field:amdsmi_pwr_t.power"], {"unit": ["W"]})
        self.assertEqual(sem["field:amdsmi_pwr_t.flags"], {"encoding": ["bitmask"]})
        self.assertEqual(sem["param:amdsmi_get_pwr.limit"], {"dir": "out", "unit": ["W"]})
        self.assertEqual(sem["param:amdsmi_get_pwr.info"], {"dir": "out"})

    def test_changed_unit_blocks(self):
        self.assertEqual(
            self.paths(edit(SEM, "Socket power in W", "Socket power in mW")),
            ["semantics/field:amdsmi_pwr_t.power"],
        )
        self.assertEqual(
            self.paths(edit(SEM, "limit in W", "limit in milliwatts")),
            ["semantics/param:amdsmi_get_pwr.limit"],
        )

    def test_new_sentinel_blocks(self):
        self.assertEqual(
            self.paths(edit(SEM, "Clock in MHz", "Clock in MHz, UINT32_MAX when unavailable")),
            ["semantics/field:amdsmi_pwr_t.clk"],
        )

    def test_signed_and_unsigned_maxima_differ(self):
        unsigned = edit(SEM, "Clock in MHz", "Clock in MHz, UINT32_MAX when unavailable")
        signed = edit(SEM, "Clock in MHz", "Clock in MHz, INT32_MAX when unavailable")
        self.assertEqual(
            abi_gate.snapshot_text(signed)["semantics"]["field:amdsmi_pwr_t.clk"]["sentinel"],
            ["0x7fffffff"],
        )
        self.assertEqual(self.paths(signed, old=unsigned), ["semantics/field:amdsmi_pwr_t.clk"])

    def test_renamed_parameter_keeps_its_meaning(self):
        renamed = edit(SEM, "uint32_t* limit);", "uint32_t* max_limit);")
        self.assertEqual(self.paths(edit(renamed, "] limit limit", "] max_limit limit")), [])
        self.assertEqual(
            self.paths(edit(renamed, "] limit limit in W", "] max_limit limit in mW")),
            ["semantics/param:amdsmi_get_pwr.limit"],
        )

    def test_rewording_or_dropping_meaning_does_not_block(self):
        self.assertEqual(
            self.paths(edit(SEM, "Socket power in W", "Power of the socket, in watts")), []
        )
        self.assertEqual(self.paths(edit(SEM, "Clock in MHz", "Clock")), [])

    def test_direction_into_the_library_blocks_but_narrowing_does_not(self):
        self.assertEqual(
            self.paths(edit(SEM, "@param[out] info", "@param[in,out] info")),
            ["semantics/param:amdsmi_get_pwr.info"],
        )
        self.assertEqual(
            self.paths(edit(SEM, "//!< Feature bitmask", "//!< IN: feature bitmask")),
            ["semantics/field:amdsmi_pwr_t.flags"],
        )
        widened = edit(SEM, "@param[out] info", "@param[in,out] info")
        self.assertEqual(self.paths(SEM, old=widened), [])

    def test_direction_aliases(self):
        inout = edit(SEM, "@param[out] info", "@param[inout] info")
        self.assertEqual(self.paths(inout), ["semantics/param:amdsmi_get_pwr.info"])
        self.assertEqual(
            self.paths(edit(SEM, "@param[out] info", "@param[in,out] info"), old=inout), []
        )
        output = edit(SEM, "//!< Feature bitmask", "//!< OUTPUT: feature bitmask")
        self.assertEqual(
            self.paths(
                edit(SEM, "//!< Feature bitmask", "//!< INPUT: feature bitmask"), old=output
            ),
            ["semantics/field:amdsmi_pwr_t.flags"],
        )


@unittest.skipUnless(HAVE_LIBCLANG, "libclang Python bindings not installed")
class ConsistencyTest(unittest.TestCase):
    def test_current_shape_is_consistent(self):
        self.assertEqual(abi_gate.consistency_errors(BASE), [])

    def test_cplusplus_only_member(self):
        text = (
            PRELUDE
            + "typedef struct { uint32_t a;\n#ifdef __cplusplus\n uint32_t b;\n#endif\n} amdsmi_c_t;\n"
        )
        errors = abi_gate.consistency_errors(text)
        self.assertTrue(any("amdsmi_c_t" in e and "C++17" in e for e in errors), errors)

    def test_cplusplus_only_field_type(self):
        text = edit(
            BASE,
            "uint32_t version;",
            "#ifdef __cplusplus\n float version;\n#else\n uint32_t version;\n#endif\n",
        )
        errors = abi_gate.consistency_errors(text)
        self.assertTrue(any("amdsmi_info_t" in e and "float" in e for e in errors), errors)

    def test_functions_must_match_and_keep_c_linkage_in_cplusplus(self):
        dropped = edit(
            BASE,
            "amdsmi_status_t amdsmi_old_api(void);",
            "#ifndef __cplusplus\namdsmi_status_t amdsmi_old_api(void);\n#endif\n",
        )
        errors = abi_gate.consistency_errors(dropped)
        self.assertTrue(any("amdsmi_old_api" in e for e in errors), errors)
        errors = abi_gate.consistency_errors(edit(edit(BASE, EXTERN_C, ""), END_EXTERN_C, ""))
        self.assertTrue(any("amdsmi_get_info" in e and "linkage" in e for e in errors), errors)

    def test_esmi_only_layout(self):
        text = PRELUDE + (
            "#ifdef ENABLE_ESMI_LIB\ntypedef struct { uint32_t a; } amdsmi_e_t;\n"
            "#else\ntypedef struct { uint64_t a; } amdsmi_e_t;\n#endif\n"
        )
        errors = abi_gate.consistency_errors(text)
        self.assertTrue(any("without ENABLE_ESMI_LIB" in e for e in errors), errors)

    def test_long_differs_on_llp64(self):
        errors = abi_gate.consistency_errors(
            PRELUDE + "typedef struct { unsigned long a; } amdsmi_l_t;\n"
        )
        self.assertTrue(any("LLP64" in e for e in errors), errors)


@unittest.skipUnless(HAVE_LIBCLANG, "libclang Python bindings not installed")
class DocTest(unittest.TestCase):
    def test_direction_and_doc_changes(self):
        new = edit(
            BASE, "//!< Output: layout version", "//!< IN: requested layout. OUT: layout filled"
        )
        new = edit(new, "@param[out] info", "@param[in,out] info")
        notes = abi_gate.doc_changes(abi_gate.snapshot_text(BASE), abi_gate.snapshot_text(new))
        self.assertIn(["amdsmi_get_info.info", "direction", "[out]", "[in,out]"], notes)
        self.assertIn(["amdsmi_info_t.version", "direction", "-", "IN/OUT"], notes)

    def test_trailing_comment_after_wrapped_declaration(self):
        text = (
            PRELUDE
            + "typedef struct {\n  uint32_t list\n      [4];  //!< first line\n            //!< second line\n} amdsmi_w_t;\n"
        )
        snap = abi_gate.snapshot_text(text)
        self.assertEqual(snap["docs"]["field:amdsmi_w_t.list"], "first line second line")

    def test_new_entities_are_not_reported(self):
        new = BASE + "typedef struct { uint32_t a; //!< doc\n} amdsmi_new_t;\n"
        self.assertEqual(
            abi_gate.doc_changes(abi_gate.snapshot_text(BASE), abi_gate.snapshot_text(new)), []
        )


class AcknowledgedTest(unittest.TestCase):
    PATH = "types/amdsmi_info_t/size"

    def entry(self, old, new, path=PATH):
        return {"path": path, "old": old, "new": new, "why": "#1"}

    def test_exact_old_and_new(self):
        brk = abi_gate.Break(self.PATH.split("/"), 48, 56, [])
        self.assertTrue(abi_gate.acknowledged(brk, [self.entry(48, 56)]))
        self.assertFalse(abi_gate.acknowledged(brk, [self.entry(48, 64)]))
        self.assertFalse(abi_gate.acknowledged(brk, [self.entry(40, 56)]))
        self.assertFalse(abi_gate.acknowledged(brk, [self.entry(48, 56, "types/x_t/size")]))

    def test_entries_chain_across_prs(self):
        brk = abi_gate.Break(self.PATH.split("/"), 48, 64, [])
        self.assertTrue(abi_gate.acknowledged(brk, [self.entry(56, 64), self.entry(48, 56)]))
        self.assertFalse(abi_gate.acknowledged(brk, [self.entry(56, 64)]))

    def test_removal_and_retired_values(self):
        path = "types/amdsmi_e_t/enumerators/E_B"
        brk = abi_gate.Break(path.split("/"), 1, abi_gate.MISSING, [])
        exc = abi_gate.load_exceptions(None)
        exc["breaks"].append({**brk.entry(), "why": "#2"})
        self.assertIsNone(exc["breaks"][0]["new"])
        self.assertTrue(abi_gate.acknowledged(brk, exc["breaks"]))
        self.assertEqual(abi_gate.policy_of(exc)["retired"], {"amdsmi_e_t": {1}})

    def test_render_round_trips_one_entry_per_line(self):
        exc = abi_gate.load_exceptions(None)
        exc["append_only"]["amdsmi_x_t"] = "struct_size header"
        exc["breaks"] += [self.entry(48, 56), self.entry([8, 4, "int"], None)]
        text = abi_gate.render_exceptions(exc)
        self.assertEqual(abi_gate.load_exceptions(text), exc)
        self.assertEqual(sum('"path"' in line for line in text.splitlines()), 2)

    def test_malformed_entry_is_refused(self):
        with self.assertRaises(abi_gate.GateError):
            abi_gate.load_exceptions('{"breaks": [{"path": "macros/X", "old": 1, "new": 2}]}')


@unittest.skipUnless(HAVE_LIBCLANG and shutil.which("git"), "needs libclang and git")
class GateTest(unittest.TestCase):
    """End to end on a throwaway repository laid out like rocm-systems."""

    BREAK = "# Changelog\n- ABI break: amdsmi_info_t gained extra\n"

    def setUp(self):
        self.repo = tempfile.mkdtemp(prefix="abi_gate_test_")
        self.git("init", "-q")
        self.git("config", "user.email", "t@example.com")
        self.git("config", "user.name", "t")
        self.write(abi_gate.HEADER, BASE)
        self.write(abi_gate.CHANGELOG, "# Changelog\n")
        self.write_exceptions(abi_gate.load_exceptions(None))
        self.commit("base")
        self.base = self.rev()

    def tearDown(self):
        shutil.rmtree(self.repo)

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", self.repo, *args], check=True, capture_output=True, text=True
        ).stdout

    def write(self, rel, text):
        path = os.path.join(self.repo, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

    def exceptions(self):
        with open(os.path.join(self.repo, abi_gate.EXCEPTIONS)) as f:
            return abi_gate.load_exceptions(f.read())

    def write_exceptions(self, exc):
        self.write(abi_gate.EXCEPTIONS, abi_gate.render_exceptions(exc))

    def acknowledge(self, release=None):
        return abi_gate.acknowledge(self.repo, self.base, release, "#1 test")

    def commit(self, msg):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", msg)

    def rev(self):
        return self.git("rev-parse", "HEAD").strip()

    def run_gate(self, release=None):
        return abi_gate.gate(
            self.repo, self.base, "HEAD", release, compile_check=bool(shutil.which("gcc"))
        )

    def grow(self, extra="uint64_t extra;"):
        self.write(abi_gate.HEADER, edit(BASE, "uint64_t value;", f"uint64_t value; {extra}"))

    def add_api(self):
        with open(os.path.join(self.repo, abi_gate.HEADER)) as f:
            text = f.read()
        self.write(
            abi_gate.HEADER, edit(text, END_EXTERN_C, "int amdsmi_more(void);\n" + END_EXTERN_C)
        )

    def unacked(self, result):
        return {(b["ref"], b["path"]) for b in result["breaks"] if not b["acknowledged"]}

    def test_no_change_passes(self):
        self.assertFalse(self.run_gate()["blocking"])

    def test_break_blocks_without_acknowledgement(self):
        self.grow()
        self.commit("grow")
        result = self.run_gate()
        self.assertTrue(result["blocking"])
        self.assertIn(("base", "types/amdsmi_info_t/size"), self.unacked(result))

    def test_acknowledgement_with_changelog_passes(self):
        self.grow()
        added = self.acknowledge()
        self.assertIn(
            {"path": "types/amdsmi_info_t/size", "old": 48, "new": 56, "why": "#1 test"}, added
        )
        self.write(abi_gate.CHANGELOG, self.BREAK)
        self.commit("grow, acknowledged")
        result = self.run_gate()
        self.assertFalse(result["blocking"], result)
        self.assertIn("types/amdsmi_info_t/size", result["acknowledged_new"])
        self.assertEqual(self.acknowledge(), [])

    def test_acknowledgement_without_changelog_blocks(self):
        self.grow()
        self.acknowledge()
        self.commit("grow, no changelog")
        result = self.run_gate()
        self.assertTrue(result["blocking"])
        self.assertTrue(any("CHANGELOG" in e for e in result["errors"]))

    def test_changelog_must_call_it_a_break(self):
        self.grow()
        self.acknowledge()
        self.write(abi_gate.CHANGELOG, "# Changelog\n- Added extra to amdsmi_info_t\n")
        self.commit("grow, changelog does not say ABI")
        errors = self.run_gate()["errors"]
        self.assertTrue(any("'ABI' or 'breaking'" in e for e in errors), errors)

    def test_entry_for_another_value_does_not_acknowledge(self):
        self.grow()
        exc = self.exceptions()
        exc["breaks"].append(
            {"path": "types/amdsmi_info_t/size", "old": 48, "new": 64, "why": "#1"}
        )
        self.write_exceptions(exc)
        self.write(abi_gate.CHANGELOG, self.BREAK)
        self.commit("acknowledge a different size")
        self.assertIn(("base", "types/amdsmi_info_t/size"), self.unacked(self.run_gate()))

    def test_entry_already_on_the_base_does_not_acknowledge(self):
        self.grow()
        self.acknowledge()
        self.write(abi_gate.HEADER, BASE)
        self.commit("acknowledge ahead of the change")
        self.base = self.rev()
        self.grow()
        self.write(abi_gate.CHANGELOG, self.BREAK)
        self.commit("grow under the old entry")
        self.assertIn(("base", "types/amdsmi_info_t/size"), self.unacked(self.run_gate()))
        with self.assertRaisesRegex(abi_gate.GateError, "already has this entry"):
            self.acknowledge()
        abi_gate.acknowledge(self.repo, self.base, None, "#2 lands it again")
        self.commit("acknowledge again")
        self.assertFalse(self.run_gate()["blocking"])

    def test_removal_needs_an_entry(self):
        self.write(abi_gate.HEADER, edit(BASE, "amdsmi_status_t amdsmi_old_api(void);", ""))
        self.write(abi_gate.CHANGELOG, "# Changelog\n- Removed amdsmi_old_api (breaking)\n")
        self.commit("remove without an entry")
        self.assertTrue(self.run_gate()["blocking"])
        self.acknowledge()
        self.commit("acknowledge")
        self.assertFalse(self.run_gate()["blocking"])

    def test_release_breaks_stay_acknowledged_and_chain(self):
        release = self.base
        self.grow()
        self.acknowledge()
        self.write(abi_gate.CHANGELOG, self.BREAK)
        self.commit("acknowledged break")
        self.base = self.rev()
        self.grow()
        self.add_api()
        self.commit("later additive change")
        result = self.run_gate(release=release)
        self.assertFalse(result["blocking"], result)
        self.assertTrue(any(b["ref"] == "release" for b in result["breaks"]))
        self.base = self.rev()
        self.grow("uint64_t extra; uint64_t more;")
        self.add_api()
        self.acknowledge(release)
        self.write(abi_gate.CHANGELOG, self.BREAK + "- ABI break: amdsmi_info_t gained more\n")
        self.commit("second break")
        result = self.run_gate(release=release)
        self.assertFalse(result["blocking"], result)
        self.assertEqual(
            [e["new"] for e in self.exceptions()["breaks"] if e["path"].endswith("/size")], [56, 64]
        )

    def test_policy_added_in_the_same_pr_does_not_exempt_the_break(self):
        self.grow()
        exc = self.exceptions()
        exc["append_only"]["amdsmi_info_t"] = "caller passes a size"
        self.write_exceptions(exc)
        self.commit("exempt the type and grow it")
        result = self.run_gate()
        self.assertTrue(result["blocking"])
        self.assertTrue(any("amdsmi_info_t" in e for e in result["errors"]), result["errors"])
        self.assertIn("policy/amdsmi_info_t", result["acknowledged_new"])

    def test_semantic_change_needs_an_entry_and_a_changelog_line(self):
        self.write(
            abi_gate.HEADER, edit(BASE, "//!< Output: layout version", "//!< Version in MHz")
        )
        self.commit("unit")
        result = self.run_gate()
        self.assertTrue(result["blocking"])
        self.assertEqual(
            {p for _, p in self.unacked(result)}, {"semantics/field:amdsmi_info_t.version"}
        )
        self.acknowledge()
        self.write(abi_gate.CHANGELOG, "# Changelog\n- ABI: amdsmi_info_t version is in MHz\n")
        self.commit("acknowledge")
        self.assertFalse(self.run_gate()["blocking"])

    def test_note_needs_a_changelog_line(self):
        exc = self.exceptions()
        exc["notes"]["field:amdsmi_info_t.value"] = "0 means unsupported"
        self.write_exceptions(exc)
        self.commit("annotate by hand")
        errors = self.run_gate()["errors"]
        self.assertTrue(any("amdsmi_info_t" in e for e in errors), errors)
        self.write(
            abi_gate.CHANGELOG, "# Changelog\n- ABI: amdsmi_info_t value 0 means unsupported\n"
        )
        self.commit("changelog")
        self.assertFalse(self.run_gate()["blocking"])

    def test_retired_enumerator_value_cannot_be_reused_later(self):
        self.write(abi_gate.HEADER, edit(BASE, "AMDSMI_STATUS_INVAL = 1,", ""))
        self.acknowledge()
        self.write(abi_gate.CHANGELOG, "# Changelog\n- Removed AMDSMI_STATUS_INVAL (breaking)\n")
        self.commit("remove")
        self.assertFalse(self.run_gate()["blocking"])
        self.base = self.rev()
        self.write(
            abi_gate.HEADER, edit(BASE, "AMDSMI_STATUS_INVAL = 1,", "AMDSMI_STATUS_NEW = 1,")
        )
        self.commit("reuse the value")
        result = self.run_gate()
        self.assertTrue(result["blocking"])
        self.assertIn(
            ("base", "types/amdsmi_status_t/enumerators/AMDSMI_STATUS_NEW"), self.unacked(result)
        )

    def test_dropping_an_enumerator_entry_blocks(self):
        self.write(abi_gate.HEADER, edit(BASE, "AMDSMI_STATUS_INVAL = 1,", ""))
        self.acknowledge()
        self.write(abi_gate.CHANGELOG, "# Changelog\n- Removed AMDSMI_STATUS_INVAL (breaking)\n")
        self.commit("remove")
        self.base = self.rev()
        self.write_exceptions(abi_gate.load_exceptions(None))
        self.commit("forget the retired value")
        errors = self.run_gate()["errors"]
        self.assertTrue(any("AMDSMI_STATUS_INVAL" in e for e in errors), errors)

    def test_acknowledged_kind_change_passes(self):
        self.write(
            abi_gate.HEADER,
            edit(
                BASE,
                "typedef struct {\n  uint32_t version;",
                "typedef union {\n  uint32_t version;",
            ),
        )
        self.acknowledge()
        self.write(abi_gate.CHANGELOG, "# Changelog\n- ABI break: amdsmi_info_t is a union\n")
        self.commit("struct to union")
        result = self.run_gate()
        self.assertFalse(result["blocking"], result)

    def test_end_sentinel_growth_passes_the_gate(self):
        self.write(
            abi_gate.HEADER,
            edit(BASE, "AMDSMI_BLOCK_LAST = 2", "AMDSMI_X = 2, AMDSMI_BLOCK_LAST = 3"),
        )
        self.commit("append")
        result = self.run_gate()
        self.assertFalse(result["blocking"], result)

    def test_base_is_the_merge_base(self):
        fork = self.base
        self.add_api()
        self.commit("develop adds an API")
        self.base = self.rev()
        self.git("checkout", "-q", "-b", "topic", fork)
        self.write(abi_gate.CHANGELOG, "# Changelog\n- unrelated\n")
        self.commit("topic")
        self.assertFalse(self.run_gate()["blocking"])
        self.assertEqual(self.acknowledge(), [])

    def test_header_shadowing_a_shim_blocks(self):
        self.write(os.path.join(os.path.dirname(abi_gate.HEADER), "stdint.h"), "typedef int x;\n")
        self.commit("shadow stdint.h")
        self.assertTrue(any("shadows" in e for e in self.run_gate()["errors"]))

    @unittest.skipUnless(shutil.which("gcc"), "needs gcc")
    def test_asserts_catch_layout_change_in_c(self):
        self.grow()
        self.commit("grow")
        result = self.run_gate()
        self.assertTrue(any("sizeof(amdsmi_info_t)" in e for e in result["errors"]))
        self.assertFalse(any("offsetof(amdsmi_info_t, value)" in e for e in result["errors"]))


if __name__ == "__main__":
    unittest.main()
