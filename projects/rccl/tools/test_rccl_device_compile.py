#!/usr/bin/env python3
"""Unit tests for the --emit-device-obj mode added to tools/rccl-device-compile.

Only covers what this change actually touches: --emit-device-obj recognition
in parse_compiler_flags() and the mode-dispatch wiring in main(). The rest of
rccl-device-compile (assembly extraction, resource aggregation, dispatcher
patching) predates this change and is untouched by it. do_emit_device_obj
itself is a thin amdclang++ subprocess wrapper like do_compile/do_link, so
(like those) it's exercised by the required clean build rather than mocked
here.

rccl-device-compile has no .py suffix (it's installed as a CMake custom-
language driver), so it's loaded here via importlib rather than a normal
import.
"""

import importlib.machinery
import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "rccl-device-compile")

# No .py suffix means importlib can't pick a loader for it by extension alone
# -- hand it the source-file loader explicitly.
_loader = importlib.machinery.SourceFileLoader("rccl_device_compile", MODULE_PATH)
_spec = importlib.util.spec_from_loader("rccl_device_compile", _loader)
rdc = importlib.util.module_from_spec(_spec)
_loader.exec_module(rdc)


class EmitDeviceObjFlagParsingTest(unittest.TestCase):
    def test_emit_device_obj_flag_is_recognized_as_our_flag(self):
        our_args, forwarded, sources = rdc.parse_compiler_flags([
            "--emit-device-obj", "--arch=gfx950", "-DFOO", "-Ipath",
            "-o", "out.o", "in.cu.cpp",
        ])
        self.assertIn("--emit-device-obj", our_args)
        self.assertEqual(forwarded, ["-DFOO", "-Ipath"])
        self.assertEqual(sources, ["in.cu.cpp"])

    def test_emit_device_obj_coexists_with_other_our_flags(self):
        our_args, _, _ = rdc.parse_compiler_flags([
            "--emit-device-obj", "--arch=gfx942", "--clang", "/opt/rocm/bin/amdclang++",
            "--keep-temps", "-o", "out.o", "in.cu.cpp",
        ])
        self.assertEqual(
            our_args,
            ["--emit-device-obj", "--arch=gfx942", "--clang", "/opt/rocm/bin/amdclang++",
             "--keep-temps", "-o", "out.o"],
        )


class EmitDeviceObjModeDispatchTest(unittest.TestCase):
    """main() requires --arch/-o/a source for --emit-device-obj, same as --compile."""

    def _run_main_with(self, argv):
        old_argv = rdc.sys.argv
        rdc.sys.argv = ["rccl-device-compile"] + argv
        try:
            rdc.main()
        finally:
            rdc.sys.argv = old_argv

    def test_missing_arch_is_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            self._run_main_with(["--emit-device-obj", "-o", "out.o", "in.cu.cpp"])
        self.assertIn("--arch", str(cm.exception))

    def test_missing_output_is_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            self._run_main_with(["--emit-device-obj", "--arch=gfx950", "in.cu.cpp"])
        self.assertIn("-o", str(cm.exception))

    def test_missing_source_is_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            self._run_main_with(["--emit-device-obj", "--arch=gfx950", "-o", "out.o"])
        self.assertIn("source file", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
