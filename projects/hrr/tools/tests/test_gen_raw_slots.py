# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""An archive integer reaches HIP as a pointer or a handle only
through translation.

Regenerates the playback handlers with gen_hrr_api_args.py and fails if any
handler still casts an archive field (``a->field``) straight to a pointer or an
opaque handle type. Also checks the other generator rules this ticket adds: a
fixed-size string is copied into a bounded, NUL-terminated local, hipLinkAddFile
is refused, and an option array with a pointer-carrying JIT option is refused.

The generator writes its capture and header outputs under projects/clr and
projects/hrr/include by default; this test sends all three outputs to a scratch
directory so it never writes into the tree.

Run: python3 -m unittest discover -s projects/hrr/tools/tests
"""

import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
GEN = TOOLS / "gen_hrr_api_args.py"
COMMITTED_PLAYBACK = TOOLS.parent / "playback" / "hip_playback_generated.cpp"
MANUAL_PLAYBACK = TOOLS.parent / "playback" / "hip_playback.cpp"

# `(Type)a->field`: a cast applied directly to a field of the recorded struct.
_CAST_OF_FIELD = re.compile(r"\((?P<type>[^()]*?)\)\s*a->(?P<field>\w+)")
# `static_cast<Type>(a->field)` and `reinterpret_cast<Type>(a->field)`.
_NAMED_CAST_OF_FIELD = re.compile(r"(?:static|reinterpret)_cast<(?P<type>[^<>]*?)>\(\s*a->(?P<field>\w+)\s*\)")
# A pointer-typed cast of a recorded field in hand-written code: the archive's
# integer used as an address. Inline `_bytes` arrays are data, not addresses.
_RAW_POINTER_CAST_MANUAL = re.compile(
    r"(?:reinterpret_cast<(?:const )?[\w:]+\s*\*+>\(\s*a->(?P<f1>\w+)\s*\)|\((?:const )?[\w:]+\s*\*+\)\s*a->(?P<f2>\w+))")


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_hrr_api_args", GEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_hrr_api_args"] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load_generator()

# Types a raw cast of an archive integer must never produce: opaque handles
# (hipDevice_t is an int ordinal, not one) and function pointers.
_POINTER_HANDLE_TYPES = (set(gen._HANDLE_TYPES) | set(gen._NON_CASTABLE_TYPES)) - {"hipDevice_t"}


def raw_slot_casts(text):
    """Return [(type, field)] for every cast of a recorded field to a pointer
    or handle type. A field that ends in _bytes is inline data in the record,
    not an address, and is not a violation."""
    found = []
    for rx in (_CAST_OF_FIELD, _NAMED_CAST_OF_FIELD):
        for m in rx.finditer(text):
            t = re.sub(r"\b(const|volatile|struct|enum)\b", "", m.group("type")).strip()
            t = re.sub(r"\s+", " ", t)
            if m.group("field").endswith("_bytes"):
                continue
            if "*" in t or t in _POINTER_HANDLE_TYPES:
                found.append((m.group("type").strip(), m.group("field")))
    return found


def manual_raw_pointer_casts(text):
    """Hand-written handlers: pointer casts of a recorded field (other than the
    inline _bytes arrays), including `reinterpret_cast<void*>(a->dst)`. Casts of
    the payload itself (`reinterpret_cast<const hrr_args_X*>(pl)`) do not use
    `a->` and are not matched."""
    out = []
    for m in _RAW_POINTER_CAST_MANUAL.finditer(text):
        field = m.group("f1") or m.group("f2")
        if not field.endswith("_bytes"):
            out.append((m.group(0), field))
    return out


def _handler_body(text, api):
    m = re.search(r"static hipError_t playback_%s\(PlaybackContext& ctx.*?\n\}\n" % re.escape(api), text, re.S)
    return m.group(0) if m else ""


class RawSlotRule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        out = Path(cls._tmp.name)
        proc = subprocess.run(
            [sys.executable, str(GEN), "--silent",
             "--output-header", str(out / "hrr_api_args.h"),
             "--output-capture", str(out / "capture.cpp"),
             "--output-playback", str(out / "playback.cpp")],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("generator failed:\n" + proc.stdout + proc.stderr)
        cls.playback = (out / "playback.cpp").read_text()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_regenerated_handlers_have_no_raw_pointer_or_handle_cast(self):
        bad = raw_slot_casts(self.playback)
        self.assertEqual(bad, [], "generated playback casts an archive field to a pointer/handle: %r" % bad[:10])

    def test_committed_handlers_have_no_raw_pointer_or_handle_cast(self):
        bad = raw_slot_casts(COMMITTED_PLAYBACK.read_text())
        self.assertEqual(bad, [], "committed hip_playback_generated.cpp casts an archive field to a pointer/handle: %r" % bad[:10])

    def test_manual_handlers_do_not_use_a_recorded_field_as_an_address(self):
        bad = manual_raw_pointer_casts(MANUAL_PLAYBACK.read_text())
        self.assertEqual(bad, [], "hip_playback.cpp casts a recorded field to a pointer: %r" % bad[:10])

    def test_attribute_union_with_a_window_pointer_is_refused(self):
        body = _handler_body(self.playback, "hipGraphKernelNodeSetAttribute")
        self.assertIn("a->attr == hipKernelNodeAttributeAccessPolicyWindow", body)
        self.assertIn("return hipErrorNotSupported;", body)
        self.assertLess(body.index("AccessPolicyWindow"), body.index("(hipError_t)hipGraphKernelNodeSetAttribute("))

    def test_scanner_detects_the_patterns_it_guards_against(self):
        self.assertTrue(raw_slot_casts("x = static_cast<const textureReference*>(a->tex);"))
        self.assertTrue(raw_slot_casts("x = reinterpret_cast<hipExecutionCtx_t>(a->ctx);"))
        self.assertTrue(manual_raw_pointer_casts("void* dst = reinterpret_cast<void*>(a->dst);"))
        self.assertTrue(manual_raw_pointer_casts("auto p = (const char*)a->name;"))
        self.assertFalse(manual_raw_pointer_casts("auto p = reinterpret_cast<const char*>(a->name_bytes);"))
        self.assertFalse(manual_raw_pointer_casts("const auto* a = reinterpret_cast<const hrr_args_X*>(pl);"))
        # If the scanner stopped matching, the two tests above would pass for
        # the wrong reason.
        self.assertTrue(raw_slot_casts("x = (const textureReference*)a->tex;"))
        self.assertTrue(raw_slot_casts("x = (hipExecutionCtx_t)a->ctx;"))
        self.assertTrue(raw_slot_casts("x = (hipStreamCallback_t)a->callback;"))
        self.assertFalse(raw_slot_casts("x = (size_t)a->size; y = (hipDevice_t)a->device;"))
        self.assertFalse(raw_slot_casts("x = (const char*)a->path_bytes;"))

    def test_pointer_and_handle_slots_raise(self):
        for raw, kind in (("const textureReference*", "pointer"),
                          ("hipExecutionCtx_t", "handle"),
                          ("hipGraphicsResource_t", "handle"),
                          ("hipStreamCallback_t", "handle")):
            self.assertEqual(gen._raw_slot_kind(raw), kind, raw)
        for raw in ("size_t", "unsigned short", "int", "hipMemcpyKind", "hipDevice_t",
                    "unsigned long long", "enum hipLimit_t", "float", "uint64_t",
                    "enum hipFlushGPUDirectRDMAWritesTarget", "enum hipFlushGPUDirectRDMAWritesScope"):
            self.assertIsNone(gen._raw_slot_kind(raw), raw)

    def test_refused_apis_are_refused_by_name(self):
        for api in ("hipImportExternalMemory", "hipBindTexture", "hipExecutionCtxSynchronize",
                    "hipLibraryGetGlobal", "hipLinkAddFile"):
            body = _handler_body(self.playback, api)
            self.assertIn("hrr_note_unreplayable", body, api)
            self.assertIn("return hipErrorNotSupported;", body, api)
            self.assertNotIn("a->", body.split("hrr_note_unreplayable")[0].replace("const auto* a", ""), api)

    def test_fixed_size_strings_are_nul_terminated_locals(self):
        body = _handler_body(self.playback, "hipGetProcAddress")
        self.assertIn("static_assert(sizeof(a->symbol_bytes) == 256", body)
        self.assertIn("char _str_symbol[256]", body)
        self.assertIn("_str_symbol[sizeof(_str_symbol) - 1] = '\\0';", body)
        self.assertNotIn("(const char*)a->symbol_bytes", body)

    def test_link_create_refuses_pointer_carrying_options(self):
        body = _handler_body(self.playback, "hipLinkCreate")
        self.assertIn("hrr::jit_options_carry_pointer(_d_options, _d_options_n)", body)
        self.assertIn('hrr_note_unreplayable(ctx, "hipLinkCreate"', body)
        # The refusal has to come before the real call.
        self.assertLess(body.index("jit_options_carry_pointer"), body.index("hipLinkCreate((unsigned int)"))


class GeneratorFunctions(unittest.TestCase):
    """The new generator functions, called directly on synthetic API entries."""

    @staticmethod
    def _entry(*params, name="hipFoo"):
        return gen.ApiEntry(name, "hipError_t", [gen.Param(t, n) for t, n in params], "runtime")

    def test_raw_slot_error_names_the_argument_and_its_kind(self):
        entry = self._entry(("const textureReference*", "tex"))
        err = gen._raw_slot_error(entry, "tex", "const textureReference*", "pointer")
        self.assertIsInstance(err, gen.RawSlotError)
        self.assertIn("tex", str(err))
        self.assertIn("textureReference", str(err))
        self.assertIn("pointer", str(err))

    def test_playback_arg_raises_for_an_untranslated_pointer_or_handle(self):
        for raw in ("const textureReference*", "hipExecutionCtx_t", "hipStreamCallback_t"):
            entry = self._entry((raw, "x"))
            with self.assertRaises(gen.RawSlotError, msg=raw):
                gen._playback_arg(entry, entry.params[0], "x", [])

    def test_playback_arg_casts_a_numeric_slot(self):
        entry = self._entry(("size_t", "n"), ("unsigned short", "v"))
        self.assertEqual(gen._playback_arg(entry, entry.params[0], "n", []), "(size_t)a->n")
        self.assertEqual(gen._playback_arg(entry, entry.params[1], "v", []), "(unsigned short)a->v")

    def test_refusing_playback_shim_refuses_by_name_with_the_reason(self):
        entry = self._entry(("int", "n"))
        text = gen._refusing_playback_shim(entry, "a stated reason")
        self.assertIn("static hipError_t playback_hipFoo(", text)
        self.assertIn('hrr_note_unreplayable(ctx, "hipFoo"', text)
        self.assertIn("a stated reason", text)
        self.assertIn("return hipErrorNotSupported;", text)
        self.assertNotIn("mark_graph_incomplete", text)

    def test_refusing_playback_shim_marks_the_graph_incomplete(self):
        entry = self._entry(("hipGraph_t", "graph"), name="hipGraphAddFooNode")
        self.assertIn("mark_graph_incomplete", gen._refusing_playback_shim(entry, "r"))

    def test_generate_playback_shim_turns_a_raw_slot_into_a_refusal_and_records_it(self):
        entry = self._entry(("const textureReference*", "tex"), name="hipFooTexture")
        gen.AUTO_UNREPLAYABLE_PLAYBACK_APIS.pop("hipFooTexture", None)
        text = gen.generate_playback_shim(entry)
        self.assertIn('hrr_note_unreplayable(ctx, "hipFooTexture"', text)
        self.assertIn("return hipErrorNotSupported;", text)
        self.assertNotIn("a->tex", text.split("hrr_note_unreplayable")[0])
        self.assertIn("hipFooTexture", gen.AUTO_UNREPLAYABLE_PLAYBACK_APIS)
        gen.AUTO_UNREPLAYABLE_PLAYBACK_APIS.pop("hipFooTexture", None)

    def test_explicit_playback_only_refusal_does_not_touch_the_capture_table(self):
        # hipLinkAddFile is refused for replay only; adding it to
        # UNREPLAYABLE_PLAYBACK_APIS would also change the capture shims.
        self.assertIn("hipLinkAddFile", gen.PLAYBACK_ONLY_UNREPLAYABLE_APIS)
        self.assertNotIn("hipLinkAddFile", gen.UNREPLAYABLE_PLAYBACK_APIS)


if __name__ == "__main__":
    unittest.main()
