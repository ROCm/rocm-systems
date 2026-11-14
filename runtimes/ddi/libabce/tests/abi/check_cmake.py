#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Exercise installed and add_subdirectory ABCE consumer mode selection."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]


def run(command: list[str], *, fails: bool = False, env: dict | None = None) -> str:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env)
    if (result.returncode == 0) == fails:
        raise RuntimeError(f"unexpected result from {command}:\n{result.stdout}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--cmake", default="cmake")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="abce-consumer-") as directory:
        work = Path(directory)
        install = work / "install"
        run([args.cmake, "--install", str(args.build), "--prefix", str(install), "--component", "abce"])
        source = work / "consumer"
        source.mkdir()
        (source / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.25)
project(ABCEConsumer LANGUAGES C CXX)
find_package(ROCmRuntimeABCE 1 REQUIRED CONFIG)
foreach(language c cpp)
  add_executable(consumer_${language} consumer.${language})
  target_link_libraries(consumer_${language} PRIVATE rocm_runtime::abce)
  target_compile_definitions(consumer_${language} PRIVATE EXPECT_HEADER_ONLY=${EXPECT_HEADER_ONLY})
endforeach()
''')
        consumer = '''#include <abce/abce_host.h>
#if ABCE_ENABLE_IMPLEMENTATION != EXPECT_HEADER_ONLY
#error Consumer mode did not propagate from CMake
#endif
int main(void) {
    abce_builder_config_t config;
    abce_builder_config_initialize(&config);
    return abce_host_abi_version() != ABCE_HOST_ABI_VERSION || !config.use_copy_size_override;
}
'''
        (source / "consumer.c").write_text(consumer)
        (source / "consumer.cpp").write_text(consumer)
        environment = dict(os.environ)
        environment["PATH"] = str(install / "bin") + os.pathsep + environment.get("PATH", "")
        for mode, expected in (("ON", "1"), ("OFF", "0")):
            build = work / mode
            # Deliberately omit the mode in the ON case: default must remain ON
            # even when the package producer was configured with OFF.
            options = [] if mode == "ON" else ["-DABCE_HEADER_ONLY=OFF"]
            run([args.cmake, "-S", str(source), "-B", str(build), "-G", "Ninja",
                 f"-DCMAKE_PREFIX_PATH={install}", f"-DEXPECT_HEADER_ONLY={expected}", *options])
            run([args.cmake, "--build", str(build)])
            for language in ("c", "cpp"):
                executable = build / (f"consumer_{language}" + (".exe" if os.name == "nt" else ""))
                run([str(executable)], env=environment)
        headers = work / "headers"
        headers.mkdir()
        (headers / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.25)
project(ABCEHeaders LANGUAGES C CXX)
add_subdirectory("{(ROOT / 'api-headers').as_posix()}" api)
add_executable(consumer "{(source / 'consumer.c').as_posix()}")
target_link_libraries(consumer PRIVATE rocm_runtime::abce)
target_compile_definitions(consumer PRIVATE EXPECT_HEADER_ONLY=1)
''')
        header_build = work / "headers-build"
        run([args.cmake, "-S", str(headers), "-B", str(header_build), "-G", "Ninja"])
        run([args.cmake, "--build", str(header_build)])
        message = run([args.cmake, "-S", str(headers), "-B", str(work / "headers-off"),
                       "-G", "Ninja", "-DABCE_HEADER_ONLY=OFF"], fails=True)
        if "rocm_runtime::abce_shared" not in message:
            raise ValueError(f"missing provider was not diagnosed:\n{message}")
        header_install = work / "header-install"
        run([args.cmake, "--install", str(header_build), "--prefix", str(header_install), "--component", "abce"])
        message = run([args.cmake, "-S", str(source), "-B", str(work / "missing-provider"),
                       "-G", "Ninja", f"-DCMAKE_PREFIX_PATH={header_install}",
                       "-DABCE_HEADER_ONLY=OFF"], fails=True)
        if "requires the Rust provider" not in message:
            raise ValueError(f"missing installed provider was not diagnosed:\n{message}")
    print("Installed C/C++ ON/OFF, default ON and missing-provider CMake consumers passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
