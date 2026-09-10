import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path


RCCL_ROOT = Path(__file__).resolve().parents[4]
DRIVER_PATH = RCCL_ROOT / "tools" / "rccl-device-compile"

# The driver has no .py suffix and executes nothing at import time (its only
# module-level side effect is guarded by if __name__ == '__main__'), so loading
# it by path is safe.
_loader = SourceFileLoader("rccl_device_compile", str(DRIVER_PATH))
_spec = spec_from_loader(_loader.name, _loader)
driver = module_from_spec(_spec)
_loader.exec_module(driver)


class DropLocalNoDeadStripTest(unittest.TestCase):
    def test_drops_only_local_no_dead_strip_directives(self):
        lines = [
            "\t.globl\tkernel\n",
            "\t.no_dead_strip\t.L__profc_foo\n",
            "  .no_dead_strip .Lbar\n",
            "\t.no_dead_strip\tglobal_sym\n",   # non-local: kept
            "\t.text\n",
        ]

        result = driver._drop_local_no_dead_strip(lines)

        self.assertEqual(result, [
            "\t.globl\tkernel\n",
            "\t.no_dead_strip\tglobal_sym\n",
            "\t.text\n",
        ])


class ParseCompilerFlagsTest(unittest.TestCase):
    def test_profile_rt_joined_form_is_our_arg(self):
        our, forwarded, sources = driver.parse_compiler_flags(
            ["--link", "--profile-rt=/opt/rt/libclang_rt.profile.a", "a.o"]
        )

        self.assertIn("--profile-rt=/opt/rt/libclang_rt.profile.a", our)
        self.assertEqual(sources, ["a.o"])
        self.assertEqual(forwarded, [])

    def test_profile_rt_separate_form_consumes_value(self):
        our, forwarded, sources = driver.parse_compiler_flags(
            ["--profile-rt", "/opt/rt/lib.a", "-DFOO=1", "-Iinc", "b.o"]
        )

        self.assertEqual(our, ["--profile-rt", "/opt/rt/lib.a"])
        self.assertEqual(forwarded, ["-DFOO=1", "-Iinc"])
        self.assertEqual(sources, ["b.o"])


class BuildLinkCmdTest(unittest.TestCase):
    def test_without_profile_rt_omits_mcpu(self):
        cmd = driver.build_link_cmd("ld.lld", "device.elf", "objs.rsp",
                                    "gfx942", None)

        self.assertEqual(cmd, ["ld.lld", "-shared", "-o", "device.elf",
                               "@objs.rsp"])

    def test_with_profile_rt_appends_mcpu_and_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / "libclang_rt.profile.a"
            archive.touch()

            cmd = driver.build_link_cmd("ld.lld", "device.elf", "objs.rsp",
                                        "gfx942", str(archive))

        self.assertEqual(cmd[-2:], ["--plugin-opt=mcpu=gfx942", str(archive)])
        # mcpu must precede the archive so lld's LTO codegen targets amdgcn.
        self.assertLess(cmd.index("--plugin-opt=mcpu=gfx942"),
                        cmd.index(str(archive)))

    def test_missing_profile_rt_raises(self):
        with self.assertRaises(SystemExit):
            driver.build_link_cmd("ld.lld", "device.elf", "objs.rsp",
                                  "gfx942", "/nonexistent/lib.a")


class DispatcherCompileCmdTest(unittest.TestCase):
    def test_gline_tables_only_precedes_forwarded_g0(self):
        cmd = driver.dispatcher_compile_cmd(
            "clang", "gfx942", ["-O1", "-g0"], "disp.s", "common.cu.cpp"
        )

        self.assertIn("-gline-tables-only", cmd)
        self.assertLess(cmd.index("-gline-tables-only"), cmd.index("-g0"))

    def test_release_flags_keep_gline_tables_only(self):
        cmd = driver.dispatcher_compile_cmd(
            "clang", "gfx942", ["-O3"], "disp.s", "common.cu.cpp"
        )

        self.assertIn("-gline-tables-only", cmd)
        self.assertNotIn("-g0", cmd)


class PatchDispatcherNoDeadStripTest(unittest.TestCase):
    def test_patch_dispatcher_drops_local_no_dead_strip(self):
        lines = [
            "\t.text\n",
            "\t.no_dead_strip\t.L__profc_foo\n",
            "\t.globl\tkernel\n",
        ]

        result = driver.patch_dispatcher(lines, {})

        self.assertEqual(result, [
            "\t.text\n",
            "\t.globl\tkernel\n",
        ])


class ExtractDeviceFunctionNoDeadStripTest(unittest.TestCase):
    def test_extract_device_function_drops_local_no_dead_strip(self):
        lines = [
            "\t.type\t_Z13ncclDevFunc_fooPv, @function\n",
            "\t.no_dead_strip\t.L__profc_foo\n",
            "\t.text\n",
        ]

        extracted, resources = driver.extract_device_function(lines)

        self.assertNotIn("\t.no_dead_strip\t.L__profc_foo\n", extracted)
        self.assertEqual(resources["function_name"], "ncclDevFunc_fooPv")


if __name__ == "__main__":
    unittest.main()
