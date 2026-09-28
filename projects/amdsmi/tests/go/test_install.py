# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from run_tests import make_env


def cache_value(*, cache: str, name: str) -> str:
    prefix = name + ":"
    for line in cache.splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1]
    raise ValueError("missing CMake cache entry: " + name)


def staged_path(*, stage: Path, prefix: Path, relative: str) -> Path:
    value = Path(relative)
    absolute = value if value.is_absolute() else prefix / value
    return stage / str(absolute).lstrip("/")


def verify_install(*, project: Path, build_dir: Path) -> None:
    cache = (build_dir / "CMakeCache.txt").read_text()
    if Path(cache_value(cache=cache, name="CMAKE_HOME_DIRECTORY")).resolve() != project.resolve():
        raise ValueError("native build belongs to a different source tree")
    header = (project / "include" / "amd_smi" / "amdsmi.h").read_text()
    version = tuple(
        int(re.search(r"#define AMDSMI_LIB_VERSION_" + part + r"\s+(\d+)", header).group(1))
        for part in ("MAJOR", "MINOR")
    )
    if version != (27, 1):
        raise ValueError("this module targets the AMD SMI 27.1 header")
    prefix = Path("/amdsmi-go-stage")
    with tempfile.TemporaryDirectory(prefix="amdsmi-go-install-") as directory:
        root = Path(directory)
        stage = root / "destdir"
        install_env = os.environ.copy()
        install_env["DESTDIR"] = str(stage)
        subprocess.run(
            ["cmake", "--install", str(build_dir), "--prefix", str(prefix), "--component", "dev"],
            env=install_env,
            check=True,
        )
        data = staged_path(
            stage=stage,
            prefix=prefix,
            relative=cache_value(cache=cache, name="CMAKE_INSTALL_DATAROOTDIR"),
        )
        include = staged_path(
            stage=stage,
            prefix=prefix,
            relative=cache_value(cache=cache, name="CMAKE_INSTALL_INCLUDEDIR"),
        )
        library = staged_path(
            stage=stage,
            prefix=prefix,
            relative=cache_value(cache=cache, name="CMAKE_INSTALL_LIBDIR"),
        )
        module = data / "amd_smi" / "go"
        expected = {Path("go.mod"), Path("README.md"), Path("LICENSE")}
        for path in (project / "go").rglob("*.go"):
            relative = path.relative_to(project / "go")
            if (
                "testdata" not in relative.parts
                and not path.name.startswith("mock_")
                and not path.name.endswith("_test.go")
            ):
                expected.add(relative)
        actual = {path.relative_to(module) for path in module.rglob("*") if path.is_file()}
        if actual != expected:
            raise ValueError("installed Go source set differs: " + str(actual ^ expected))
        for relative in expected:
            if (module / relative).read_bytes() != (project / "go" / relative).read_bytes():
                raise ValueError("installed file differs: " + str(relative))
        if (module / "LICENSE").read_bytes() != (project / "LICENSE").read_bytes():
            raise ValueError("nested module license differs from project license")
        if not (include / "amd_smi" / "amdsmi.h").is_file():
            raise FileNotFoundError("staged public header missing")
        if (include / "amd_smi" / "amdsmi.h").read_bytes() != (
            project / "include" / "amd_smi" / "amdsmi.h"
        ).read_bytes():
            raise ValueError("staged header differs from the intended source header")
        if not (library / "libamd_smi.so").is_file():
            raise FileNotFoundError("staged native library missing")
        env = make_env(
            root=root, include_dir=include, library_dir=library, cc="cc", cgocheck2=False
        )
        subprocess.run(["go", "build", "./..."], cwd=module, env=env, check=True)
        consumer = root / "consumer"
        consumer.mkdir()
        module_path = "github.com/ROCm/rocm-systems/projects/amdsmi/go"
        (consumer / "go.mod").write_text(
            "module example.com/amdsmi-install-check\n\ngo 1.20\n\n"
            + "require "
            + module_path
            + " v0.0.0\n"
            + "replace "
            + module_path
            + " => "
            + json.dumps(str(module))
            + "\n"
        )
        (consumer / "main.go").write_text(
            'package main\nimport smi "' + module_path + '/amdsmi"\n'
            "func main() { v, err := smi.GetLibraryVersion(); "
            "if err != nil { panic(err) }; "
            'if v.Major != 27 || v.Minor != 1 { panic("native version mismatch") } }\n'
        )
        binary = root / "native-version"
        subprocess.run(["go", "build", "-o", str(binary), "."], cwd=consumer, env=env, check=True)
        subprocess.run([str(binary)], env=env, check=True)
        print(
            "Verified {} installed Go files and staged AMD SMI 27.1 consumer".format(len(expected))
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify staged AMD SMI Go development sources")
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify_install(
            project=Path(__file__).resolve().parents[2], build_dir=args.build_dir.resolve()
        )
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
