# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Access to rocprofiler-sdk's avail library and its Python module.

Owns everything about reaching `librocprofv3-list-avail.so`: where it lives on
a given ROCm install, how it is loaded, and how its C entry points are called.
Callers get plain Python values and never see ctypes or a library path.
"""

import ctypes
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from utils.logger import console_debug, console_error
from utils.utils_common import resolve_rocm_library_path

LIBRARY_NAME = "librocprofv3-list-avail.so"

# Set on every load: the library caches its PC sampling enumeration on first
# use and dlopen will not re-init it, so an ungated first load is unrecoverable.
_BETA_GATE = "ROCPROFILER_PC_SAMPLING_BETA_ENABLED"

# Number of out-parameters pc_sample_config() fills in.
_CONFIG_FIELD_COUNT = 5

_library: Optional[ctypes.CDLL] = None
_library_path: Optional[str] = None


def get_pc_sample_configs(
    sdk_tool_path: Optional[str],
) -> Optional[list[tuple[int, ...]]]:
    """Return (method, unit, min, max, flags) for every agent configuration.

    None means the library could not be reached at all, which is distinct from
    an empty list: agents were queried and reported no configuration.
    """
    library = _load_library(sdk_tool_path)
    if library is None:
        return None

    configs = []
    for agent_handle in _agent_handles(library):
        configs.extend(_agent_configs(library, agent_handle))
    return configs


def get_counters(sdk_tool_path: Optional[str]) -> dict[str, Any]:
    """Return the avail module's per-agent counter listing."""
    # Load through ctypes first so the beta gate is applied before the avail
    # module dlopens the same file and settles what the library reports.
    _load_library(sdk_tool_path)
    if _library_path is None:
        console_error(f"Failed to locate {LIBRARY_NAME}.")

    avail = _import_avail_module(sdk_tool_path)
    avail.loadLibrary.libname = _library_path
    return avail.get_counters()


def resolve_library_path(sdk_tool_path: Optional[str]) -> Optional[str]:
    """Locate the avail library for the ROCm install the tool belongs to.

    The library moved between ROCm versions:
      ROCm >= 7.1: <rocm_path>/lib/rocprofiler-sdk/
      ROCm 7.0.x:  <rocm_path>/libexec/rocprofiler-sdk/
    """
    if not sdk_tool_path:
        return None

    candidates = (
        Path(sdk_tool_path).parent / LIBRARY_NAME,
        Path(sdk_tool_path).parents[2] / "libexec" / "rocprofiler-sdk" / LIBRARY_NAME,
    )
    for candidate in candidates:
        resolved = resolve_rocm_library_path(str(candidate))
        if resolved and Path(resolved).exists():
            return resolved

    console_debug(f"{LIBRARY_NAME} not found for {sdk_tool_path}")
    return None


def _load_library(sdk_tool_path: Optional[str]) -> Optional[ctypes.CDLL]:
    """Load the avail library once per process, with PC sampling enabled."""
    global _library, _library_path
    if _library is not None:
        return _library

    library_path = resolve_library_path(sdk_tool_path)
    if library_path is None:
        return None

    previous_gate = os.environ.get(_BETA_GATE)
    os.environ[_BETA_GATE] = "ON"
    try:
        library = ctypes.CDLL(library_path)
        _bind_signatures(library)
        # Forces the enumeration while the gate is in place. Without this call
        # the gate has no effect, since the library reads it on first use.
        library.get_number_of_agents()
    except OSError as err:
        console_debug(f"Unable to load {library_path}: {err}")
        return None
    finally:
        _restore_gate(previous_gate)

    _library, _library_path = library, library_path
    return library


def _restore_gate(previous_gate: Optional[str]) -> None:
    """Put the beta gate back so it never reaches the profiled application."""
    if previous_gate is None:
        os.environ.pop(_BETA_GATE, None)
    else:
        os.environ[_BETA_GATE] = previous_gate


def _bind_signatures(library: ctypes.CDLL) -> None:
    """Declare return and argument types for the entry points used here."""
    library.get_number_of_agents.restype = ctypes.c_ulong
    library.get_number_of_pc_sample_configs.restype = ctypes.c_ulong
    library.pc_sample_config.argtypes = [ctypes.c_ulong, ctypes.c_ulong] + [
        ctypes.POINTER(ctypes.c_ulong)
    ] * _CONFIG_FIELD_COUNT


def _agent_handles(library: ctypes.CDLL) -> list[int]:
    """Return a handle for every agent the library knows about."""
    agent_count = library.get_number_of_agents()
    library.agent_handles.argtypes = [ctypes.c_ulong * agent_count, ctypes.c_ulong]
    handles = (ctypes.c_ulong * agent_count)()
    library.agent_handles(handles, agent_count)
    return list(handles)


def _agent_configs(library: ctypes.CDLL, agent_handle: int) -> list[tuple[int, ...]]:
    """Marshal one agent's PC sampling configurations out of the library."""
    configs = []
    for config_index in range(library.get_number_of_pc_sample_configs(agent_handle)):
        fields = [ctypes.c_ulong() for _ in range(_CONFIG_FIELD_COUNT)]
        library.pc_sample_config(
            agent_handle, config_index, *(ctypes.byref(field) for field in fields)
        )
        configs.append(tuple(field.value for field in fields))
    return configs


def _import_avail_module(sdk_tool_path: Optional[str]) -> ModuleType:
    """Import the avail Python module shipped alongside the tool.

    The module moved from <rocm_path>/bin/rocprofv3_avail_module/avail.py to
    <rocm_path>/lib/python3/site-packages/rocprofv3/avail.py.
    """
    new_path = str(Path(sdk_tool_path).parents[1] / "python3/site-packages")
    old_path = str(Path(sdk_tool_path).parents[2] / "bin")
    try:
        sys.path.append(new_path)
        from rocprofv3 import avail

        return avail
    except ImportError:
        console_debug(
            f"Could not import rocprofiler-sdk avail module from {new_path}, "
            f"trying {old_path}"
        )

    try:
        sys.path.remove(new_path)
        sys.path.append(old_path)
        from rocprofv3_avail_module import avail

        return avail
    except ImportError:
        # console_error exits, so this branch never returns.
        console_error("Failed to import rocprofiler-sdk avail module.")
