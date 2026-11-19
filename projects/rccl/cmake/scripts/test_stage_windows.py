"""Source-staging regression tests; no HIP compiler, GPU, Bash or Perl required."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("rccl_stage_windows", Path(__file__).with_name("stage_windows.py"))
stage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage
SPEC.loader.exec_module(stage)


class StageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rccl staging test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.output, self.sdk = (self.root / name for name in ("source", "output", "sdk"))
        self.write("cmake/scripts/add_unroll.sh", "unroll script")
        self.write("cmake/scripts/add_faults.sh", "fault script")
        self.tools = stage.Tools(*(self.root / name for name in ("perl.exe", "bash.exe", "hipify-perl")))
        for tool in (self.tools.perl, self.tools.bash, self.tools.hipify):
            tool.write_text("test tool", encoding="utf-8")
        (self.sdk / ".info").mkdir(parents=True)
        (self.sdk / ".info/version").write_text("10.2.0", encoding="utf-8")
        self.commands = []

    def write(self, relative, content):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def fake_run(self, command, environment):
        self.commands.append(command)
        if command[0] == self.tools.perl:
            command[-1].write_text(command[-3].read_text(encoding="utf-8"), encoding="utf-8")
        else:
            output = Path(command[-1])
            output.write_text(output.read_text(encoding="utf-8") + "\n// transformed", encoding="utf-8")

    def build(self, **kwargs):
        with mock.patch.object(stage, "run", side_effect=self.fake_run), contextlib.redirect_stdout(io.StringIO()):
            stage.stage_sources(self.source, self.output, self.sdk, self.tools, jobs=1, **kwargs)

    def test_cache_uses_content_even_if_source_mtime_is_preserved(self):
        source = self.write("src/main.cc", "first")
        self.build()
        count = len(self.commands)
        self.build()
        self.assertEqual(len(self.commands), count)
        times = source.stat()
        source.write_text("second", encoding="utf-8")
        os.utime(source, ns=(times.st_atime_ns, times.st_mtime_ns))
        self.build()
        self.assertEqual((self.output / "src/main.cc").read_text(), "second")
        self.assertEqual(len(self.commands), count + 1)

    def test_tool_sdk_and_unroll_changes_invalidate_cache(self):
        self.write("src/device/common.h", "device header")
        self.build()
        for dependency in (self.tools.hipify, self.tools.perl, self.tools.bash,
                           self.source / "cmake/scripts/add_unroll.sh", self.sdk / ".info/version"):
            count = len(self.commands)
            dependency.write_text(dependency.read_text() + " changed", encoding="utf-8")
            self.build()
            self.assertEqual(len(self.commands), count + 2, str(dependency))

    def test_failed_transform_preserves_previous_output_and_retries(self):
        source = self.write("src/device/common.h", "old")
        self.build()
        output = self.output / "src/device/common.h"
        old_output = output.read_bytes()
        old_state = (self.output / stage.STATE_FILE).read_bytes()
        source.write_text("new", encoding="utf-8")

        def fail_after_hipify(command, environment):
            self.fake_run(command, environment)
            if command[0] == self.tools.bash:
                raise stage.StagingError("unroll failed")

        with mock.patch.object(stage, "run", side_effect=fail_after_hipify):
            with self.assertRaisesRegex(stage.StagingError, "unroll failed"):
                stage.stage_sources(self.source, self.output, self.sdk, self.tools, jobs=1)
        self.assertEqual(output.read_bytes(), old_output)
        self.assertEqual((self.output / stage.STATE_FILE).read_bytes(), old_state)
        self.assertEqual(list(output.parent.glob(".common.*.h")), [])
        self.build()
        self.assertTrue(output.read_text().startswith("new"))

    def test_deleted_and_renamed_sources_remove_only_managed_outputs(self):
        original = self.write("src/old.cc", "old")
        self.build()
        unrelated = self.output / "src/generated.h"
        unrelated.write_text("generated elsewhere", encoding="utf-8")
        original.rename(original.with_name("new.cc"))
        self.build()
        self.assertFalse((self.output / "src/old.cc").exists())
        self.assertTrue((self.output / "src/new.cc").is_file())
        self.assertEqual(unrelated.read_text(), "generated elsewhere")

    def test_raw_compatibility_headers_are_not_hipified(self):
        for name in stage.COPY_UNCHANGED:
            self.write(name, "#ifdef __CUDACC__\n// CUDA and HIP branches\n#endif\n")
        self.build()
        self.assertEqual(self.commands, [])
        for name in stage.COPY_UNCHANGED:
            self.assertEqual((self.source / name).read_bytes(), (self.output / name).read_bytes())

    def test_include_aliases_preserve_genuine_or_unknown_tmp_headers(self):
        unit = self.write("src/main.cc", "")
        self.write("src/include/param/common.h", "")
        self.write("src/include/genuine_tmp.h", "")
        self.write("src/include/genuine.h", "")
        text = '#include "param/common_tmp.h"\n#include "genuine_tmp.h"\n#include <external_tmp.h>\n'
        self.assertEqual(stage.normalize_collision_includes(text, unit, self.source),
                         '#include "param/common.h"\n#include "genuine_tmp.h"\n#include <external_tmp.h>\n')

    def test_fault_injection_and_dependencies_are_recorded(self):
        self.write("src/device/common.h", "device header")
        stamp = self.root / "stage stamp"
        depfile = self.root / "stage.d"
        self.build(fault_injection=True, stamp=stamp, depfile=depfile)
        self.assertEqual(len(self.commands), 3)
        self.assertEqual(self.commands[-1][1], self.source / "cmake/scripts/add_faults.sh")
        self.assertTrue(stamp.is_file())
        dependencies = depfile.read_text()
        for path in (self.tools.hipify, self.source / "cmake/scripts/add_faults.sh", self.source / "src/device/common.h"):
            self.assertIn(stage.depfile_escape(path), dependencies)
        self.assertIn("\\ ", dependencies)

    def test_output_damage_forces_a_restage(self):
        self.write("src/main.cc", "correct")
        self.build()
        (self.output / "src/main.cc").write_text("corrupt", encoding="utf-8")
        self.build()
        self.assertEqual((self.output / "src/main.cc").read_text(), "correct")

    def test_cleanup_rejects_paths_outside_staging(self):
        for relative in ("../outside.cc", "src/../../outside.cc"):
            with self.assertRaises(stage.StagingError):
                stage.managed_output(self.output, relative)


class ToolDiscoveryTests(unittest.TestCase):
    def test_windows_bash_launchers_do_not_override_native_git_bash(self):
        with tempfile.TemporaryDirectory() as temporary:
            native = Path(temporary) / "bash.exe"
            native.touch()
            launcher = Path(temporary) / "Microsoft/WindowsApps/bash.EXE"
            with mock.patch.object(stage.shutil, "which", return_value=str(launcher)):
                self.assertEqual(stage.resolve_tool(None, ("bash",), [native]), native.resolve())

    def test_git_install_can_be_discovered_outside_program_files(self):
        with tempfile.TemporaryDirectory(prefix="custom git ") as temporary:
            root = Path(temporary)
            git = root / "portable-git/cmd/git.exe"
            git.parent.mkdir(parents=True)
            git.touch()
            tools_dir = git.parent.parent / "usr/bin"
            tools_dir.mkdir(parents=True)
            for name in ("bash.exe", "perl.exe"):
                (tools_dir / name).touch()
            sdk = root / "sdk"
            hipify = sdk / "libexec/hipify/hipify-perl"
            hipify.parent.mkdir(parents=True)
            hipify.touch()
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                    stage.shutil, "which", side_effect=lambda name: str(git) if name in ("git", "git.exe") else None):
                tools = stage.discover_tools(sdk)
            self.assertEqual(tools.perl, (tools_dir / "perl.exe").resolve())
            self.assertEqual(tools.bash, (tools_dir / "bash.exe").resolve())
            self.assertEqual(tools.hipify, hipify.resolve())

    def test_explicit_tool_override_wins_over_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            override = Path(temporary) / "custom-perl.exe"
            override.touch()
            with mock.patch.object(stage.shutil, "which", return_value=None):
                self.assertEqual(stage.resolve_tool(override, ("perl",)), override.resolve())
                with self.assertRaisesRegex(stage.StagingError, "Requested tool"):
                    stage.resolve_tool(override.with_name("missing.exe"), ("perl",))

    def test_external_tool_error_includes_stderr(self):
        result = mock.Mock(returncode=4, stdout="", stderr="missing dependency")
        with mock.patch.object(stage.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(stage.StagingError, "missing dependency"):
                stage.run(["perl", "hipify"], {})


if __name__ == "__main__":
    unittest.main()
