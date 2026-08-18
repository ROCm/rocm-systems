# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility guard between the vendored nccl4py sources and the HIP shim.

The vendored ``nccl`` package imports cuda.core directly, and on ROCm those
imports are served by ``nccl._hip_compat.cuda_core_shim`` rather than a real
cuda.core. cuda.core has spelled its stream protocol and device-pointer alias
differently in every generation, so a sync that brings a drop using a new
spelling breaks ``import nccl.core`` unless the shim grows it. The required
surface is therefore derived from the vendored sources themselves.

Only the symbol-resolution test needs hip-python; namespace registration is
HIP-free, which keeps the guard meaningful on a CPU-only host.
"""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from pathlib import Path

import pytest

import nccl  # noqa: F401  (registers the cuda.core shim under sys.modules)

_PKG_DIR = Path(nccl.__file__).parent
_SHIM_DIR = _PKG_DIR / "_hip_compat"

# Both prefixes must resolve to the same module objects.
_ALIASED_SUFFIXES = (
    "",
    ".system",
    ".utils",
    ".typing",
    "._stream",
    "._memory",
    "._memory._buffer",
)


@dataclass(frozen=True)
class _CudaCoreImport:
    """A single ``cuda.core`` import found in the vendored sources."""

    module: str
    name: str | None
    where: str

    def __str__(self) -> str:
        target = self.module if self.name is None else f"{self.module}.{self.name}"
        return f"{target} ({self.where})"


def _is_cuda_core(module: str | None) -> bool:
    return module is not None and (module == "cuda.core" or module.startswith("cuda.core."))


def _scan_cuda_core_imports() -> tuple[_CudaCoreImport, ...]:
    """Collect every cuda.core import in the vendored package.

    The shim is excluded: it serves these imports, so scanning it would compare
    the shim against itself.
    """
    found: list[_CudaCoreImport] = []
    for path in sorted(_PKG_DIR.rglob("*.py")):
        if _SHIM_DIR in path.parents:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = path.relative_to(_PKG_DIR.parent)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and _is_cuda_core(node.module):
                for alias in node.names:
                    found.append(
                        _CudaCoreImport(node.module, alias.name, f"{rel}:{node.lineno}")
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_cuda_core(alias.name):
                        found.append(
                            _CudaCoreImport(alias.name, None, f"{rel}:{node.lineno}")
                        )
    return tuple(found)


_VENDORED_IMPORTS = _scan_cuda_core_imports()


class TestVendoredImportSurface:
    def test_scan_finds_imports(self):
        """Guards the other two tests from passing on an empty scan."""
        assert _VENDORED_IMPORTS, f"no cuda.core imports found under {_PKG_DIR}"

    def test_every_imported_module_resolves(self):
        unresolved = []
        for imp in _VENDORED_IMPORTS:
            try:
                importlib.import_module(imp.module)
            except ImportError as exc:
                unresolved.append(f"{imp} -> {exc}")
        assert not unresolved, "cuda.core modules missing from the HIP shim:\n" + "\n".join(
            unresolved
        )

    def test_every_imported_symbol_resolves(self):
        pytest.importorskip("hip", reason="hip-python is required to resolve HIP-backed symbols")
        missing = []
        for imp in _VENDORED_IMPORTS:
            if imp.name is None:
                continue
            module = importlib.import_module(imp.module)
            if not hasattr(module, imp.name):
                missing.append(str(imp))
        assert not missing, "symbols missing from the HIP shim:\n" + "\n".join(missing)


class TestTypingSurface:
    def test_exports_cuda_core_1_0_names(self):
        typing_mod = importlib.import_module("cuda.core.typing")
        assert typing_mod.IsStreamType is not None
        assert typing_mod.DevicePointerType is not None

    def test_pre_1_0_names_are_aliases(self):
        typing_mod = importlib.import_module("cuda.core.typing")
        assert typing_mod.IsStreamT is typing_mod.IsStreamType
        assert typing_mod.DevicePointerT is typing_mod.DevicePointerType

    @pytest.mark.parametrize(
        "module_name, attr",
        [
            ("cuda.core", "IsStreamType"),
            ("cuda.core", "DevicePointerType"),
            ("cuda.core._stream", "IsStreamT"),
            ("cuda.core._memory", "DevicePointerT"),
            ("cuda.core._memory._buffer", "DevicePointerT"),
        ],
    )
    def test_pre_1_0_locations_serve_the_same_objects(self, module_name, attr):
        typing_mod = importlib.import_module("cuda.core.typing")
        module = importlib.import_module(module_name)
        assert getattr(module, attr) is getattr(typing_mod, attr)


class TestExperimentalAlias:
    @pytest.mark.parametrize("suffix", _ALIASED_SUFFIXES)
    def test_prefix_aliases_the_top_level(self, suffix):
        experimental = importlib.import_module(f"cuda.core.experimental{suffix}")
        assert experimental is importlib.import_module(f"cuda.core{suffix}")

    def test_pre_0_5_0_import_paths_resolve(self):
        from cuda.core.experimental._memory import DevicePointerT
        from cuda.core.experimental._stream import IsStreamT
        from cuda.core.experimental.typing import DevicePointerType, IsStreamType

        assert IsStreamT is IsStreamType
        assert DevicePointerT is DevicePointerType


class TestOutOfScopeAttributes:
    def test_unknown_top_level_attribute_raises(self):
        import cuda.core

        with pytest.raises(AttributeError, match="HIP shim"):
            cuda.core.LaunchConfig

    def test_unknown_typing_attribute_raises(self):
        typing_mod = importlib.import_module("cuda.core.typing")

        with pytest.raises(AttributeError, match="HIP shim"):
            typing_mod.StreamType
