# SPDX-License-Identifier: MIT

"""Host-only regressions for the Cargo/CMake boundary (no Rust build required)."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cargo_artifacts import Artifact, collect_messages, copy_changed
from configure_rust import cmake_set, rust_flags, validate_host


class ConfigureTests(unittest.TestCase):
    def test_cargo_config_paths_survive_working_directory_changes(self):
        helpers = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "cmake").symlink_to(helpers, target_is_directory=True)
            (source / "rust-toolchain.toml").write_text(
                '[toolchain]\nchannel = "1.98.0"\n'
            )
            (source / "Cargo.lock").write_text("# fixture\n")
            config = root / "config with spaces.toml"
            config.write_text("# fixture config\n")
            alias = root / "config alias.toml"
            alias.symlink_to(config)
            # Fake provisioned tools validate actual subprocess arguments; no Rust
            # build is needed to exercise configure, generated commands or Ninja.
            for tool in ("cargo", "rustc", "ld.lld"):
                executable = root / tool
                executable.write_text(f"""#!{sys.executable}
import json
from pathlib import Path
import sys
tool = Path(sys.argv[0]).name
if tool == "rustc":
    print("rustc 1.98.0\\nrelease: 1.98.0\\nhost: x86_64-unknown-linux-gnu")
elif tool == "ld.lld":
    print("LLD fixture")
elif "--version" in sys.argv:
    print("cargo 1.98.0")
else:
    if "--config" in sys.argv:
        config = Path(sys.argv[sys.argv.index("--config") + 1])
        assert config.is_absolute() and config.is_file(), str(config)
    print(json.dumps({{"packages": [], "workspace_members": []}}))
""")
                executable.chmod(0o755)
            (source / "CMakeLists.txt").write_text(
                """cmake_minimum_required(VERSION 3.25)
project(ConfigFixture LANGUAGES NONE)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_C_COMPILER "unused compiler")
set(CMAKE_BUILD_TYPE Release)
include(cmake/RuntimeRust.cmake)
runtime_rust_initialize()
file(APPEND "${CMAKE_BINARY_DIR}/configure-count" "configured\\n")
add_custom_target(check ALL COMMAND ${_runtime_cargo_command} metadata --frozen
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" VERBATIM)
"""
            )
            for index, value in enumerate(
                (config.name, str(config), alias.name, "", "missing.toml")
            ):
                with self.subTest(config=value):
                    binary = root / f"build-{index}"
                    result = subprocess.run(
                        [
                            "cmake",
                            "-S",
                            str(source),
                            "-B",
                            str(binary),
                            "-G",
                            "Ninja",
                            f"-DROCM_RUNTIMES_CARGO={root / 'cargo'}",
                            f"-DROCM_RUNTIMES_RUSTC={root / 'rustc'}",
                            f"-DROCM_RUNTIMES_LLD={root / 'ld.lld'}",
                            f"-DROCM_RUNTIMES_CARGO_CONFIG:FILEPATH={value}",
                        ],
                        cwd=root,
                        capture_output=True,
                        text=True,
                    )
                    if value == "missing.toml":
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(
                            "Cargo configuration does not exist", result.stderr
                        )
                        continue
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    settings = (binary / "runtime-rust-config.cmake").read_text()
                    if value:
                        expected = os.path.abspath(root / value)
                        self.assertIn(
                            cmake_set("_runtime_cargo_config", [expected]), settings
                        )
                        self.assertIn(
                            f"ROCM_RUNTIMES_CARGO_CONFIG:FILEPATH={expected}",
                            (binary / "CMakeCache.txt").read_text(),
                        )
                    else:
                        self.assertNotIn('"--config"', settings)
                    subprocess.run(
                        ["cmake", "--build", str(binary)],
                        cwd=source,
                        check=True,
                        capture_output=True,
                    )
                    if value:
                        before = (binary / "configure-count").read_text()
                        config.write_text(config.read_text() + "# changed\n")
                        subprocess.run(
                            ["cmake", "--build", str(binary)],
                            cwd=source,
                            check=True,
                            capture_output=True,
                        )
                        self.assertEqual(
                            (binary / "configure-count").read_text(),
                            before + "configured\n",
                        )

    def test_library_registration_uses_project_version_and_platform_guard(self):
        module = Path(__file__).resolve().parents[1] / "RuntimeRust.cmake"
        for system in ("Linux", "Windows", "Darwin"):
            with self.subTest(
                system=system
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                script = f"""cmake_minimum_required(VERSION 3.25)
project(Fixture VERSION 7.2 LANGUAGES NONE)
include("{module.as_posix()}")
set(CMAKE_SYSTEM_NAME "{system}")
set(CMAKE_SHARED_LIBRARY_PREFIX lib)
set(CMAKE_SHARED_LIBRARY_SUFFIX .so)
set(_runtime_library_dir "${{CMAKE_BINARY_DIR}}/lib")
add_custom_target(rocm_runtime_rust)
runtime_rust_library(fixture PACKAGE unrelated LIBRARY unrelated TYPE SHARED
  OUTPUT_NAME unrelated VERSION "${{PROJECT_VERSION}}" SOVERSION "${{PROJECT_VERSION_MAJOR}}"
  SONAME_ENV FIXTURE_SONAME)
get_target_property(location fixture IMPORTED_LOCATION)
get_target_property(soname fixture IMPORTED_SONAME)
get_property(environment GLOBAL PROPERTY _runtime_soname_env)
get_property(links GLOBAL PROPERTY _runtime_link_commands)
file(WRITE "${{CMAKE_BINARY_DIR}}/result" "${{location}}\\n${{soname}}\\n${{environment}}\\n${{links}}\\n")
"""
                (root / "CMakeLists.txt").write_text(script)
                result = subprocess.run(
                    ["cmake", "-S", str(root), "-B", str(root / "build")],
                    capture_output=True,
                    text=True,
                )
                if system == "Linux":
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    lines = (root / "build/result").read_text().splitlines()
                    self.assertTrue(lines[0].endswith("/lib/libunrelated.so.7.2"))
                    self.assertEqual(
                        lines[1:3],
                        ["libunrelated.so.7", "FIXTURE_SONAME=libunrelated.so.7"],
                    )
                    self.assertIn("create_symlink;libunrelated.so.7.2;", lines[3])
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(f"not implemented for {system}", result.stderr)

    def test_native_host_architectures(self):
        for processor, host in (
            ("AMD64", "x86_64-unknown-linux-gnu"),
            ("aarch64", "aarch64-unknown-linux-gnu"),
            ("ppc64le", "powerpc64le-unknown-linux-gnu"),
            ("ppc64", "powerpc64-unknown-linux-gnu"),
            ("riscv64", "riscv64gc-unknown-linux-gnu"),
        ):
            with self.subTest(processor=processor):
                validate_host(processor, host)
        for processor, host in (
            ("ppc64le", "powerpc64-unknown-linux-gnu"),
            ("riscv64", "x86_64-unknown-linux-gnu"),
            ("unknown", "unknown-linux-gnu"),
        ):
            with self.subTest(processor=processor), self.assertRaises(ValueError):
                validate_host(processor, host)

    def test_rust_flags_preserve_arguments_and_encoded_precedence(self):
        compiler = "/tools with spaces/gcc"
        flags = rust_flags(
            {"RUSTFLAGS": "--cfg 'label=\"two words\"'"}, compiler
        ).split("\x1f")
        self.assertEqual(flags[:2], ["--cfg", 'label="two words"'])
        self.assertIn(f"linker={compiler}", flags)
        flags = rust_flags(
            {"RUSTFLAGS": "ignored", "CARGO_ENCODED_RUSTFLAGS": "-C\x1fopt-level=2"},
            compiler,
        )
        self.assertTrue(flags.startswith("-C\x1fopt-level=2\x1f"))
        self.assertNotIn("ignored", flags)
        self.assertNotIn(
            "ignored",
            rust_flags(
                {"RUSTFLAGS": "ignored", "CARGO_ENCODED_RUSTFLAGS": ""}, compiler
            ),
        )

    def test_cmake_argument_roundtrip(self):
        values = [
            "path with spaces",
            "semi;colon",
            "${do_not_expand}",
            'quote"and\\slash',
            "a\x1fb",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "arguments.cmake"
            script.write_text(
                cmake_set("values", values)
                + 'foreach(value IN LISTS values)\nfile(APPEND "${CMAKE_CURRENT_LIST_DIR}/result" "${value}\\n")\nendforeach()\n'
            )
            subprocess.run(
                ["cmake", "-P", str(script)], check=True, capture_output=True
            )
            self.assertEqual((root / "result").read_text().splitlines(), values)


class ArtifactTests(unittest.TestCase):
    def test_artifact_identity_and_test_filtering(self):
        artifact = Artifact("workspace#first", "first", ".so", Path("staged.so"))

        def message(package, test, filenames):
            return json.dumps(
                {
                    "reason": "compiler-artifact",
                    "package_id": package,
                    "target": {"name": "first"},
                    "profile": {"test": test},
                    "filenames": filenames,
                }
            )

        output = "\n".join(
            [
                message("external#first", False, ["wrong.so"]),
                message("workspace#first", True, ["test.so"]),
                message("workspace#first", False, ["right.so", "right.a"]),
            ]
        )
        found, _ = collect_messages(output, [artifact])
        self.assertEqual(found, {artifact: Path("right.so")})
        with self.assertRaisesRegex(RuntimeError, "Expected one"):
            collect_messages(
                message("workspace#first", False, ["a.so", "b.so"]), [artifact]
            )

    def test_copy_changed_preserves_mtime_and_rejects_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            destination = Path(directory) / "destination"
            source.write_bytes(b"artifact")
            copy_changed(source, destination)
            timestamp = destination.stat().st_mtime_ns
            copy_changed(source, destination)
            self.assertEqual(destination.stat().st_mtime_ns, timestamp)
            source.write_bytes(b"")
            with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                copy_changed(source, destination)

    def test_multiple_static_packages_and_failure_are_isolated(self):
        helper = Path(__file__).resolve().parents[1] / "cargo_artifacts.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "workspace_members": ["first", "second"],
                        "packages": [
                            {"name": name, "id": name} for name in ("first", "second")
                        ],
                    }
                )
            )
            command = [sys.executable, str(helper), "--metadata", str(metadata)]
            messages = []
            for name, libraries in (("first", "-lm -ldl -lm"), ("second", "-lpthread")):
                source = root / f"{name}.a"
                source.write_bytes(name.encode())
                command += [
                    "--artifact",
                    name,
                    name,
                    ".a",
                    str(root / "stage" / source.name),
                    "--native-libs",
                    name,
                    name,
                    str(root / f"{name}.rsp"),
                ]
                messages += [
                    {
                        "reason": "compiler-artifact",
                        "package_id": name,
                        "target": {"name": name},
                        "profile": {"test": False},
                        "filenames": [str(source)],
                    },
                    {
                        "reason": "compiler-message",
                        "package_id": name,
                        "target": {"name": name},
                        "message": {"message": f"native-static-libs: {libraries}"},
                    },
                ]
            output = "\n".join(json.dumps(message) for message in messages)
            emit = ["--", sys.executable, "-c", f"print({output!r})"]
            subprocess.run(command + emit, check=True, capture_output=True)
            self.assertEqual((root / "first.rsp").read_text(), "-lm -ldl -lm\n")
            self.assertEqual((root / "second.rsp").read_text(), "-lpthread\n")
            self.assertEqual((root / "stage/second.a").read_bytes(), b"second")
            shutil.rmtree(root / "stage")
            result = subprocess.run(
                command + emit[:-1] + [emit[-1] + "; exit(7)"], capture_output=True
            )
            self.assertEqual(result.returncode, 7)
            self.assertFalse((root / "stage").exists())
            # A missing per-package closure must fail rather than borrowing another's.
            incomplete = "\n".join(json.dumps(message) for message in messages[:-1])
            result = subprocess.run(
                command + emit[:-1] + [f"print({incomplete!r})"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("second/second", result.stderr)
            self.assertFalse((root / "stage").exists())


if __name__ == "__main__":
    unittest.main()
