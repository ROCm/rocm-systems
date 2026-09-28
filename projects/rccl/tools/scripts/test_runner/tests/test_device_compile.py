import argparse
import tempfile
import unittest
import unittest.mock
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


class ScratchDirTest(unittest.TestCase):
    def setUp(self):
        self._build = tempfile.TemporaryDirectory()
        self._decoy = tempfile.TemporaryDirectory()
        self.addCleanup(self._build.cleanup)
        self.addCleanup(self._decoy.cleanup)
        # Patch tempdir, not $TMPDIR: tempfile resolves it once and caches it,
        # so a regressed driver would litter the real /tmp.
        patcher = unittest.mock.patch.object(tempfile, "tempdir",
                                             self._decoy.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_scratch_dir_is_created_under_the_output_directory(self):
        obj = str(Path(self._build.name) / "sub" / "kernel.o")

        scratch = driver.make_scratch_dir(obj, "rccl-dc-")

        self.assertEqual(Path(scratch).parent, Path(self._build.name) / "sub")
        self.assertTrue(Path(scratch).is_dir())

    def test_scratch_dir_ignores_the_ambient_temp_dir(self):
        obj = str(Path(self._build.name) / "kernel.o")

        scratch = driver.make_scratch_dir(obj, "rccl-dc-")

        self.assertEqual(Path(scratch).parent, Path(self._build.name))
        self.assertEqual(list(Path(self._decoy.name).iterdir()), [])

    def test_repeated_calls_do_not_collide(self):
        obj = str(Path(self._build.name) / "kernel.o")

        first = driver.make_scratch_dir(obj, "rccl-dc-")
        second = driver.make_scratch_dir(obj, "rccl-dc-")

        self.assertNotEqual(first, second)
        # Pin the parent too, so the case fails rather than passing on
        # mkdtemp's own uniqueness if the scratch dir escapes the build tree.
        for scratch in (first, second):
            self.assertEqual(Path(scratch).parent, Path(self._build.name))

    def test_do_compile_removes_a_scratch_dir_that_is_not_empty(self):
        """rmtree rather than rmdir: a step that leaves an extra file behind
        used to leak the whole directory into the build tree."""
        def fake_run(cmd, description=""):
            asm = cmd[cmd.index('-o') + 1]
            Path(asm).write_text("")
            # An unexpected leftover, e.g. a crash dump or a .d the driver
            # does not know to name. rmdir would refuse to remove the parent.
            Path(asm).with_name("leftover.tmp").write_text("")

        out_dir = Path(self._build.name)
        args = argparse.Namespace(
            clang="clang", arch="gfx942", source="kernel.cpp",
            keep_temps=False, output=str(out_dir / "kernel.o"))

        saved = (driver.discover_tools, driver.run,
                 driver.extract_device_function)
        driver.discover_tools = lambda _: ("clang", "ld.lld")
        driver.run = fake_run
        driver.extract_device_function = lambda lines: ([], {})
        try:
            driver.do_compile(args, [])
        finally:
            (driver.discover_tools, driver.run,
             driver.extract_device_function) = saved

        self.assertEqual(list(out_dir.glob("rccl-dc-*")), [])


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


class TargetIdTest(unittest.TestCase):
    """Each case pins one side of the --arch / --target-id split, so
    collapsing them into one option fails here and not in a clean-looking
    build. The driver's module docstring explains the split.
    """

    def _main_with(self, argv, mode="compile"):
        """Run main() with do_<mode> stubbed, and return the parsed args."""
        captured = {}

        def fake(args, forwarded_flags):
            captured["args"] = args

        real = getattr(driver, f"do_{mode}")
        real_argv = driver.sys.argv
        setattr(driver, f"do_{mode}", fake)
        driver.sys.argv = ["rccl-device-compile"] + argv
        try:
            driver.main()
        finally:
            setattr(driver, f"do_{mode}", real)
            driver.sys.argv = real_argv
        return captured["args"]

    def test_joined_form_is_our_arg(self):
        our, forwarded, sources = driver.parse_compiler_flags(
            ["--compile", "--arch=gfx942", "--target-id=gfx942:xnack+",
             "-DFOO", "-o", "out.o", "in.cpp"]
        )

        self.assertIn("--target-id=gfx942:xnack+", our)
        self.assertEqual(forwarded, ["-DFOO"])
        self.assertEqual(sources, ["in.cpp"])

    def test_separate_form_consumes_value(self):
        our, forwarded, sources = driver.parse_compiler_flags(
            ["--compile", "--target-id", "gfx950:xnack+", "-O3", "in.cpp"]
        )

        self.assertEqual(our, ["--compile", "--target-id", "gfx950:xnack+"])
        self.assertEqual(forwarded, ["-O3"])
        self.assertEqual(sources, ["in.cpp"])

    def test_defaults_to_arch_when_absent(self):
        args = self._main_with(["--compile", "--arch=gfx942",
                                "-o", "out.o", "in.cpp"])

        self.assertEqual(args.target_id, "gfx942")

    def test_does_not_overwrite_an_explicit_value(self):
        args = self._main_with(["--compile", "--arch=gfx942",
                                "--target-id=gfx942:xnack+",
                                "-o", "out.o", "in.cpp"])

        self.assertEqual(args.arch, "gfx942")
        self.assertEqual(args.target_id, "gfx942:xnack+")

    def test_rejects_a_target_id_for_another_processor(self):
        with self.assertRaises(SystemExit):
            self._main_with(["--compile", "--arch=gfx942",
                             "--target-id=gfx950:xnack+",
                             "-o", "out.o", "in.cpp"])

    def test_codegen_command_builders_emit_the_id_they_are_given(self):
        compile_cmd = driver.dispatcher_compile_cmd(
            "clang", "gfx942:xnack+", [], "disp.s", "common.cu.cpp")
        assemble_cmd = driver.device_assemble_cmd(
            "clang", "gfx942:xnack+", "disp.o", "disp.s")

        self.assertIn("--offload-arch=gfx942:xnack+", compile_cmd)
        self.assertIn("-mcpu=gfx942:xnack+", assemble_cmd)

    def test_link_hands_the_target_id_to_every_codegen_step(self):
        """The builders above only prove they interpolate their argument;
        this pins do_link to reaching all three of them with --target-id,
        while ld.lld still gets the bare name. Runs the whole of do_link
        with only run() stubbed, so each command is the genuine one."""
        calls = []

        def fake_run(cmd, description=""):
            calls.append(cmd)
            if "-S" in cmd:
                Path(cmd[cmd.index("-o") + 1]).write_text("")

        with tempfile.TemporaryDirectory() as temp_dir:
            obj = Path(temp_dir) / "kernel.o"
            obj.touch()
            (Path(temp_dir) / "kernel.resources.json").write_text(
                '{"vgpr_count": 8, "agpr_count": 0, "sgpr_count": 16}')
            bitcode = Path(temp_dir) / "librocshmem_device_gfx942.bc"
            bitcode.touch()
            # build_link_cmd only emits --plugin-opt=mcpu when it is given an
            # archive, so without one the link step carries no arch at all and
            # there is nothing here to pin.
            profile_rt = Path(temp_dir) / "libclang_rt.profile.a"
            profile_rt.touch()

            args = argparse.Namespace(
                clang="clang", arch="gfx942", target_id="gfx942:xnack+",
                objects=[str(obj)], output=str(Path(temp_dir) / "device.elf"),
                dispatcher="common.cu.cpp", keep_temps=False,
                rocshmem_bitcode=str(bitcode), profile_rt=str(profile_rt))

            saved = (driver.discover_tools, driver.run)
            driver.discover_tools = lambda _: ("clang", "ld.lld")
            driver.run = fake_run
            try:
                driver.do_link(args, [])
            finally:
                driver.discover_tools, driver.run = saved

        disp_compile, disp_assemble, rocshmem_compile, link = calls
        self.assertIn("--offload-arch=gfx942:xnack+", disp_compile)
        self.assertIn("-mcpu=gfx942:xnack+", disp_assemble)
        self.assertIn("-mcpu=gfx942:xnack+", rocshmem_compile)
        # lld's LTO plugin does not parse target IDs, so this one site takes
        # the bare processor name.
        self.assertIn("--plugin-opt=mcpu=gfx942", link)
        self.assertNotIn("--plugin-opt=mcpu=gfx942:xnack+", link)

    def test_compile_hands_the_target_id_to_both_codegen_steps(self):
        """The codegen path a whole build goes through. Only run() and the
        extractor are stubbed, so both commands do_compile issues are the
        real ones."""
        calls = []

        def fake_run(cmd, description=""):
            calls.append(cmd)
            if '-S' in cmd:
                Path(cmd[cmd.index('-S') + 2]).write_text("")

        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                clang="clang", arch="gfx942", target_id="gfx942:xnack+",
                source="kernel.cpp", keep_temps=False,
                output=str(Path(temp_dir) / "kernel.o"))

            saved = (driver.discover_tools, driver.run,
                     driver.extract_device_function)
            driver.discover_tools = lambda _: ("clang", "ld.lld")
            driver.run = fake_run
            driver.extract_device_function = lambda lines: ([], {})
            try:
                driver.do_compile(args, [])
            finally:
                (driver.discover_tools, driver.run,
                 driver.extract_device_function) = saved

        compile_cmd, assemble_cmd = calls
        self.assertIn("--offload-arch=gfx942:xnack+", compile_cmd)
        self.assertIn("-mcpu=gfx942:xnack+", assemble_cmd)

    def test_lld_lto_plugin_takes_the_bare_arch(self):
        """Given a target ID, ld.lld warns "not a recognized processor" and
        links with no subtarget at all -- worse than the bare name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / "libclang_rt.profile.a"
            archive.touch()

            cmd = driver.build_link_cmd("ld.lld", "device.elf", "objs.rsp",
                                        "gfx942", str(archive))

        self.assertIn("--plugin-opt=mcpu=gfx942", cmd)

    def test_a_suffixed_arch_is_stripped(self):
        """A target ID on args.arch reaches lld and the register model, both
        of which misread it silently."""
        args = self._main_with(["--link", "--arch=gfx942:xnack+",
                                "--dispatcher=common.cu.cpp",
                                "-o", "device.elf", "kernel.o"],
                               mode="link")

        self.assertEqual(args.arch, "gfx942")
        self.assertEqual(args.target_id, "gfx942:xnack+")
        self.assertTrue(driver._has_unified_vgpr_agpr(args.arch))


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
