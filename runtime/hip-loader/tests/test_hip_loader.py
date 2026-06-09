#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import unittest


HIP_SUCCESS = 0
FORCED_ERROR = 777


CHILD_FUNCTIONAL = r"""
import ctypes
import os

HIP_SUCCESS = 0
FORCED_ERROR = 777

loader_path = os.environ["HIP_LOADER_TEST_LOADER"]
backend_path = os.environ["HIP_LOADER_TEST_BACKEND"]
mode = os.environ["HIP_LOADER_TEST_MODE"]
os.environ["HIP_LOADER_BACKEND_PATH"] = backend_path

backend = ctypes.CDLL(backend_path, mode=getattr(ctypes, "RTLD_LOCAL", 0))
backend.__testBackendReset()
loader = ctypes.CDLL(loader_path, mode=getattr(ctypes, "RTLD_LOCAL", 0))

class TestCall(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("symbol", ctypes.c_char_p),
        ("phase", ctypes.c_uint32),
        ("default_result", ctypes.c_int),
        ("args", ctypes.c_void_p * 8),
    ]

class Dim3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]

class HipMemsetNodeParamsV6(ctypes.Structure):
    _fields_ = [
        ("dst", ctypes.c_void_p),
        ("pitch", ctypes.c_size_t),
        ("value", ctypes.c_uint),
        ("elementSize", ctypes.c_uint),
        ("width", ctypes.c_size_t),
        ("height", ctypes.c_size_t),
    ]

class HipMemsetParams(ctypes.Structure):
    _fields_ = [
        ("dst", ctypes.c_void_p),
        ("elementSize", ctypes.c_uint),
        ("height", ctypes.c_size_t),
        ("pitch", ctypes.c_size_t),
        ("value", ctypes.c_uint),
        ("width", ctypes.c_size_t),
    ]

class HipDevicePropR0000Prefix(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("regsPerBlock", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsDim", ctypes.c_int * 3),
        ("maxGridSize", ctypes.c_int * 3),
        ("clockRate", ctypes.c_int),
        ("memoryClockRate", ctypes.c_int),
        ("memoryBusWidth", ctypes.c_int),
        ("totalConstMem", ctypes.c_size_t),
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("multiProcessorCount", ctypes.c_int),
        ("l2CacheSize", ctypes.c_int),
    ]

class HipDevicePropR0600Prefix(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("uuid", ctypes.c_char * 16),
        ("luid", ctypes.c_char * 8),
        ("luidDeviceNodeMask", ctypes.c_uint),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
    ]

CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(TestCall))
seen = []
forced_symbols = set()

def callback(call_ptr):
    symbol = call_ptr.contents.symbol.decode()
    seen.append(symbol)
    if symbol in forced_symbols:
        return FORCED_ERROR
    return HIP_SUCCESS

callback_ref = CALLBACK(callback)
backend.__testBackendSetAPICallback.argtypes = [CALLBACK]
backend.__testBackendSetAPICallback(callback_ref)
backend.__testBackendGetCallCount.argtypes = [ctypes.c_char_p]
backend.__testBackendGetCallCount.restype = ctypes.c_uint
backend.__testBackendGetLastMemsetParams.restype = ctypes.POINTER(HipMemsetParams)

loader.hipInit.argtypes = [ctypes.c_uint]
loader.hipInit.restype = ctypes.c_int
loader.hipRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
loader.hipRuntimeGetVersion.restype = ctypes.c_int
loader.hipDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
loader.hipDriverGetVersion.restype = ctypes.c_int
loader.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
loader.hipGetDeviceCount.restype = ctypes.c_int
loader.hipGetProcAddress.argtypes = [
    ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_int)
]
loader.hipGetProcAddress.restype = ctypes.c_int
getattr(loader, "__hipRegisterFatBinary").argtypes = [ctypes.c_void_p]
getattr(loader, "__hipRegisterFatBinary").restype = ctypes.c_void_p

def check(condition, message):
    if not condition:
        raise AssertionError(message)

check(loader.hipInit(0) == HIP_SUCCESS, "hipInit failed")

version = ctypes.c_int()
check(loader.hipRuntimeGetVersion(ctypes.byref(version)) == HIP_SUCCESS, "runtime version failed")
check(version.value == 70000000, f"unexpected runtime version {version.value}")
check(loader.hipDriverGetVersion(ctypes.byref(version)) == HIP_SUCCESS, "driver version failed")
check(version.value == 70000000, f"unexpected driver version {version.value}")

count = ctypes.c_int()
check(loader.hipGetDeviceCount(ctypes.byref(count)) == HIP_SUCCESS, "device count failed")
check(count.value == 1, "unexpected device count")

if mode == "v6":
    props_buffer = ctypes.create_string_buffer(8192)
    loader.hipGetDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.c_int]
    loader.hipGetDeviceProperties.restype = ctypes.c_int
    check(loader.hipGetDeviceProperties(ctypes.byref(props_buffer), 0) == HIP_SUCCESS, "v6 legacy props failed")
    props = HipDevicePropR0000Prefix.from_buffer(props_buffer)
    check(props.name.rstrip(b"\0") == b"example_backend generated test device", "legacy name mismatch")
    check(props.totalGlobalMem == 0x123456789, "legacy memory conversion failed")
    check(props.major == 9 and props.minor == 4, "legacy arch conversion failed")
    check(props.memoryBusWidth == 4096, "legacy memory bus conversion failed")
    check(backend.__testBackendGetCallCount(b"hipBackendV7GetDevicePropertiesR0600") == 1, "v6 did not call latest R0600 backend")
    check(backend.__testBackendGetCallCount(b"hipBackendV7GetDevicePropertiesR0000") == 0, "v6 called old backend R0000")

    old_params = HipMemsetNodeParamsV6()
    old_params.dst = 0x1000
    old_params.pitch = 32
    old_params.value = 7
    old_params.elementSize = 4
    old_params.width = 64
    old_params.height = 8
    node = ctypes.c_void_p()
    loader.hipDrvGraphAddMemsetNode.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.POINTER(HipMemsetNodeParamsV6), ctypes.c_void_p
    ]
    loader.hipDrvGraphAddMemsetNode.restype = ctypes.c_int
    check(loader.hipDrvGraphAddMemsetNode(ctypes.byref(node), None, None, 0, ctypes.byref(old_params), None) == HIP_SUCCESS, "v6 graph memset failed")
else:
    props_buffer = ctypes.create_string_buffer(8192)
    loader.hipGetDevicePropertiesR0600.argtypes = [ctypes.c_void_p, ctypes.c_int]
    loader.hipGetDevicePropertiesR0600.restype = ctypes.c_int
    check(loader.hipGetDevicePropertiesR0600(ctypes.byref(props_buffer), 0) == HIP_SUCCESS, "v7 props failed")
    props = HipDevicePropR0600Prefix.from_buffer(props_buffer)
    check(props.name.rstrip(b"\0") == b"example_backend generated test device", "R0600 name mismatch")
    check(props.totalGlobalMem == 0x123456789, "R0600 memory mismatch")
    legacy_props_buffer = ctypes.create_string_buffer(8192)
    loader.hipGetDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.c_int]
    loader.hipGetDeviceProperties.restype = ctypes.c_int
    check(loader.hipGetDeviceProperties(ctypes.byref(legacy_props_buffer), 0) == HIP_SUCCESS, "v7 legacy props failed")
    legacy_props = HipDevicePropR0000Prefix.from_buffer(legacy_props_buffer)
    check(legacy_props.totalGlobalMem == 0x123456789, "v7 legacy props conversion failed")
    check(backend.__testBackendGetCallCount(b"hipBackendV7GetDevicePropertiesR0000") == 0, "v7 called old backend R0000")

    params = HipMemsetParams()
    params.dst = 0x1000
    params.elementSize = 4
    params.height = 8
    params.pitch = 32
    params.value = 7
    params.width = 64
    node = ctypes.c_void_p()
    loader.hipDrvGraphAddMemsetNode.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.POINTER(HipMemsetParams), ctypes.c_void_p
    ]
    loader.hipDrvGraphAddMemsetNode.restype = ctypes.c_int
    check(loader.hipDrvGraphAddMemsetNode(ctypes.byref(node), None, None, 0, ctypes.byref(params), None) == HIP_SUCCESS, "v7 graph memset failed")

last = backend.__testBackendGetLastMemsetParams().contents
check(last.dst == 0x1000 and last.elementSize == 4 and last.height == 8, "memset layout conversion failed")
check(last.pitch == 32 and last.value == 7 and last.width == 64, "memset layout conversion failed")

proc = ctypes.c_void_p()
proc_status = ctypes.c_int()
check(loader.hipGetProcAddress(b"hipGetDeviceProperties", ctypes.byref(proc), 500, 0, ctypes.byref(proc_status)) == HIP_SUCCESS, "legacy proc lookup failed")
check(proc.value == ctypes.cast(loader.hipGetDeviceProperties, ctypes.c_void_p).value, "legacy proc returned wrong wrapper")
check(loader.hipGetProcAddress(b"hipGetDeviceProperties", ctypes.byref(proc), 600, 0, ctypes.byref(proc_status)) == HIP_SUCCESS, "current proc lookup failed")
check(proc.value == ctypes.cast(loader.hipGetDevicePropertiesR0600, ctypes.c_void_p).value, "current proc returned wrong wrapper")

fatbin_handle = getattr(loader, "__hipRegisterFatBinary")(ctypes.c_void_p(0xFEED))
check(fatbin_handle not in (None, 0), "private fat binary registration failed")
check(backend.__testBackendGetCallCount(b"hipBackendV7CompilerRegisterFatBinary") == 1, "compiler-private symbol was not redirected")

forced_symbols.add("hipBackendV7GetDeviceCount")
check(loader.hipGetDeviceCount(ctypes.byref(count)) == FORCED_ERROR, "forced backend error did not propagate")
check("hipBackendV7GetDeviceCount" in seen, "callback did not observe device count")
"""


CHILD_NEGATIVE = r"""
import ctypes
import os
loader_path = os.environ["HIP_LOADER_TEST_LOADER"]
backend_path = os.environ["HIP_LOADER_TEST_BACKEND"]
os.environ["HIP_LOADER_BACKEND_PATH"] = backend_path
loader = ctypes.CDLL(loader_path, mode=getattr(ctypes, "RTLD_LOCAL", 0))
loader.hipInit.argtypes = [ctypes.c_uint]
loader.hipInit.restype = ctypes.c_int
result = loader.hipInit(0)
if result == 0:
    raise AssertionError("loader accepted invalid backend")
"""


CHILD_PRELOAD = r"""
import ctypes
import os
loader_path = os.environ["HIP_LOADER_TEST_LOADER"]
backend_path = os.environ["HIP_LOADER_TEST_BACKEND"]
os.environ["HIP_LOADER_BACKEND_PATH"] = backend_path
loader = ctypes.CDLL(loader_path, mode=getattr(ctypes, "RTLD_LOCAL", 0))
loader.hipRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
loader.hipRuntimeGetVersion.restype = ctypes.c_int
version = ctypes.c_int()
result = loader.hipRuntimeGetVersion(ctypes.byref(version))
if result != 0 or version.value != 70000000:
    raise AssertionError(f"LD_PRELOAD/global namespace interposed backend symbol: result={result} version={version.value}")
"""


class HipLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--manifest", required=True, type=Path)
        parser.add_argument("--v6-loader", required=True, type=Path)
        parser.add_argument("--v7-loader", required=True, type=Path)
        parser.add_argument("--backend", required=True, type=Path)
        parser.add_argument("--wrong-major-backend", required=True, type=Path)
        parser.add_argument("--missing-handshake-backend", required=True, type=Path)
        parser.add_argument("--bad-public-export-backend", required=True, type=Path)
        parser.add_argument("--bad-compiler-export-backend", required=True, type=Path)
        parser.add_argument("--preload-interposer", required=True, type=Path)
        cls.args, _ = parser.parse_known_args()
        cls.manifest = json.loads(cls.args.manifest.read_text())
        for path in [
            cls.args.v6_loader,
            cls.args.v7_loader,
            cls.args.backend,
            cls.args.wrong_major_backend,
            cls.args.missing_handshake_backend,
            cls.args.bad_public_export_backend,
            cls.args.bad_compiler_export_backend,
            cls.args.preload_interposer,
        ]:
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"expected non-empty build artifact: {path}")

    def run_child(self, code: str, loader: Path, backend: Path, mode: str = "v7",
                  preload: Path | None = None) -> None:
        env = os.environ.copy()
        env["HIP_LOADER_TEST_LOADER"] = str(loader.resolve())
        env["HIP_LOADER_TEST_BACKEND"] = str(backend.resolve())
        env["HIP_LOADER_TEST_MODE"] = mode
        if preload is not None and sys.platform.startswith("linux"):
            env["LD_PRELOAD"] = str(preload.resolve())
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"child failed with exit code {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )

    def test_manifest_is_full_surface(self) -> None:
        public_abi = self.manifest["public_abi"]
        self.assertGreater(public_abi["6"]["export_count"], 450)
        self.assertGreater(public_abi["7"]["export_count"], 500)
        signature_sources = self.manifest["signature_sources"]
        self.assertGreater(len(signature_sources["clang_headers"]), 450)
        self.assertIn("hipInit", signature_sources["clang_headers"])
        self.assertIn("__hipPushCallConfiguration", signature_sources["clang_headers"])
        backend_functions = self.manifest["backend_header_api"]["functions"]
        self.assertIn("hipBackendV7Init", backend_functions)
        self.assertIn("hipBackendV7CompilerRegisterFatBinary", backend_functions)
        self.assertIn("hipBackendV7PrivateGetPCH", backend_functions)
        stale = set(self.manifest["stale_exports"])
        self.assertIn("hiprtcCompileProgram", stale)
        compat = {entry["symbol"] for entry in public_abi["6"]["compat_symbols"]}
        self.assertIn("hipGetDeviceProperties", compat)
        self.assertIn("hipDrvGraphAddMemsetNode", compat)

    def test_v6_loader_targets_latest_backend(self) -> None:
        self.run_child(CHILD_FUNCTIONAL, self.args.v6_loader, self.args.backend, mode="v6")

    def test_v7_loader_targets_latest_backend(self) -> None:
        self.run_child(CHILD_FUNCTIONAL, self.args.v7_loader, self.args.backend, mode="v7")

    def test_invalid_backend_is_rejected(self) -> None:
        for backend in [
            self.args.wrong_major_backend,
            self.args.missing_handshake_backend,
            self.args.bad_public_export_backend,
            self.args.bad_compiler_export_backend,
        ]:
            with self.subTest(backend=backend.name):
                self.run_child(CHILD_NEGATIVE, self.args.v7_loader, backend)

    def test_ld_preload_does_not_interpose_backend_lookup(self) -> None:
        if not sys.platform.startswith("linux"):
            self.skipTest("LD_PRELOAD is Linux-specific")
        self.run_child(CHILD_PRELOAD, self.args.v7_loader, self.args.backend,
                       preload=self.args.preload_interposer)

    def test_backend_does_not_export_public_hip_symbols(self) -> None:
        exported = self.dynamic_symbols(self.args.backend)
        self.assertIn("hipBackendV7GetInterface", exported)
        self.assertIn("hipBackendV7Init", exported)
        self.assertNotIn("hipInit", exported)
        self.assertNotIn("__hipRegisterFatBinary", exported)
        self.assertNotIn("hipGetDevicePropertiesR0000", exported)
        self.assertNotIn("hipBackendV7GetDevicePropertiesR0000", exported)

    def test_backend_has_no_public_hip_relocations(self) -> None:
        if not sys.platform.startswith("linux"):
            self.skipTest("ELF relocations are Linux-specific")
        output = self.run_readelf(["--relocations", "--wide", str(self.args.backend)])
        for line in output.splitlines():
            symbol = line.split()[-1] if line.split() else ""
            self.assertFalse(
                symbol.startswith("hip") or symbol.startswith("__hip"),
                f"backend has public HIP relocation: {line}",
            )

    def test_linux_symbol_versions_match_manifest(self) -> None:
        if not sys.platform.startswith("linux"):
            self.skipTest("ELF version nodes are Linux-specific")
        v6_symbols = self.dynamic_symbol_versions(self.args.v6_loader)
        v7_symbols = self.dynamic_symbol_versions(self.args.v7_loader)
        self.assertIn("hipGetDevicePropertiesR0000@@hip_4.2", v6_symbols)
        self.assertIn("hipGetDevicePropertiesR0600@@hip_6.0", v6_symbols)
        self.assertIn("hipDrvGraphAddMemsetNode@@hip_5.6", v6_symbols)
        self.assertNotIn("hiprtcCompileProgram", v7_symbols)
        self.assertIn("hipLaunchKernelExC@@hip_6.5", v7_symbols)
        self.assertIn("hipGetProcAddress_spt@@hip_7.2", v7_symbols)
        self.assertIn("hipProfilerEnableExt@@hip_profiler_ext", v7_symbols)
        self.assertIn("hipExecutionCtxDestroy@@hip_7.14", v7_symbols)

    def dynamic_symbols(self, library: Path) -> set[str]:
        output = self.run_readelf(["--dyn-syms", "--wide", str(library)])
        symbols: set[str] = set()
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 8:
                name = parts[-1].split("@", 1)[0]
                symbols.add(name)
        return symbols

    def dynamic_symbol_versions(self, library: Path) -> str:
        return self.run_readelf(["--dyn-syms", "--wide", str(library)])

    def run_readelf(self, args: list[str]) -> str:
        tool = self.find_elf_reader()
        completed = subprocess.run(
            [str(tool), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"{tool} failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return completed.stdout

    def find_elf_reader(self) -> Path:
        workspace_root = Path(__file__).resolve().parents[6]
        candidates = [
            workspace_root / "rocm/lib/llvm/bin/llvm-readelf",
            shutil.which("llvm-readelf"),
            shutil.which("readelf"),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            path = Path(candidate)
            if path.exists():
                return path
        raise AssertionError("could not find llvm-readelf or readelf")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
