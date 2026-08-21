# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Resolve JAX kernels to HLO operators and source locations.

Reads the optimized HLO that XLA dumps for each module and builds a lookup from
a ``(function, kernel_name)`` pair to the HLO operator, source location, and
result shape recorded in the instruction metadata.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.logger import console_debug, console_warning

# Optimized HLO text files, one per module.
_HLO_TEXT_GLOB = "*after_optimizations.txt"

# Prefix XLA adds to dumped module names.
_MODULE_NAME_PREFIX = "jit_"

# Marker operator-name prefixes.
_MARKER_PREFIXES = ("jax.jit.", "jax.pmap.")

_HLO_MODULE_RE = re.compile(r"^HloModule\s+([^\s,]+)")
_SECTION_ENTRY_RE = re.compile(r"^\s*(\d+)\s+(.*)$")
_FIELD_RE = re.compile(r"(\w+)=(\d+)")
_INSTRUCTION_RE = re.compile(r"^\s*(?:ROOT\s+)?%([\w.$-]+)\s*=\s*(.+?)\s+[\w-]+\(")
_OP_NAME_RE = re.compile(r'op_name="([^"]*)"')
_STACK_FRAME_RE = re.compile(r"stack_frame_id=(\d+)")
_TRAILING_INDEX_RE = re.compile(r"_\d+$")
_LAYOUT_RE = re.compile(r"\{[^{}]*\}")


@dataclass
class KernelResolution:
    """Attribution for one kernel.

    - ``operator_path``: operator name relative to the module function, for
      example ``jit(matmul_relu)/dot_general`` or ``reduce_sum``.
    - ``operator``: the full HLO ``op_name``.
    - ``source``: ``file:line`` of the operator, or "" when unknown.
    - ``shape``: the operator's result shape, or "" when unknown.
    """

    operator_path: str
    operator: str
    source: str
    shape: str


def normalize_kernel_name(name: str) -> str:
    """Drop a leading '%' and replace '.' with '_' to match kernel symbols."""
    return name.lstrip("%").replace(".", "_")


def _kernel_name_stem(name: str) -> str:
    """Return the normalized name without any trailing numeric suffix."""
    return _TRAILING_INDEX_RE.sub("", normalize_kernel_name(name))


def _function_from_operator(operator_name: str) -> Optional[str]:
    """Return the function name from a marker operator name.

    ``jax.jit.train_step`` -> ``train_step``. Returns None when the name is not
    a JAX marker.
    """
    for prefix in _MARKER_PREFIXES:
        if operator_name.startswith(prefix):
            return operator_name[len(prefix) :]
    return None


def _relative_operator_path(function: str, operator: str) -> str:
    """Return the operator name with the leading ``jit(<function>)`` scope removed.

    Returns the operator unchanged when that scope is absent.
    """
    module_scope = f"jit({function})/"
    if operator.startswith(module_scope):
        return operator[len(module_scope) :]
    return operator


class JaxKernelSourceMap:
    """Lookup from ``(function, kernel_name)`` to a :class:`KernelResolution`.

    ``resolve`` returns None when a kernel has no matching HLO instruction.
    """

    def __init__(self) -> None:
        # function -> normalized kernel name -> (operator, source, shape)
        self._exact: dict[str, dict[str, tuple[str, str, str]]] = {}
        # function -> kernel name stem -> set of (operator, source, shape)
        self._by_stem: dict[str, dict[str, set[tuple[str, str, str]]]] = {}

    def is_empty(self) -> bool:
        return not self._exact

    def _add(
        self,
        function: str,
        instruction_name: str,
        operator: str,
        source: str,
        shape: str,
    ) -> None:
        normalized = normalize_kernel_name(instruction_name)
        # Keep the first entry, preferring one that has a source location.
        exact = self._exact.setdefault(function, {})
        if normalized not in exact or (not exact[normalized][1] and source):
            exact[normalized] = (operator, source, shape)
        stem = self._by_stem.setdefault(function, {}).setdefault(
            _kernel_name_stem(instruction_name), set()
        )
        stem.add((operator, source, shape))

    def resolve(
        self, operator_name: str, kernel_name: str
    ) -> Optional[KernelResolution]:
        """Resolve a marker operator and kernel name to a KernelResolution."""
        function = _function_from_operator(operator_name)
        if function is None or function not in self._exact:
            return None

        entry = self._exact[function].get(normalize_kernel_name(kernel_name))
        if entry is None:
            # Fall back to the name stem when the match is unambiguous.
            candidates = self._by_stem[function].get(_kernel_name_stem(kernel_name))
            if not candidates or len(candidates) != 1:
                return None
            entry = next(iter(candidates))

        operator, source, shape = entry
        return KernelResolution(
            operator_path=_relative_operator_path(function, operator),
            operator=operator,
            source=source,
            shape=shape,
        )


def _parse_indexed_section(lines: list[str], start: int) -> tuple[dict[int, str], int]:
    """Parse a ``<id> <value>`` section, returning the map and the next index."""
    entries: dict[int, str] = {}
    index = start
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            break
        match = _SECTION_ENTRY_RE.match(line)
        if not match:
            break
        entries[int(match.group(1))] = match.group(2).strip()
        index += 1
    return entries, index


def _build_source_table(
    file_names: dict[int, str],
    file_locations: dict[int, str],
    stack_frames: dict[int, str],
) -> dict[int, str]:
    """Map each stack_frame_id to a ``file:line`` string."""
    location_source: dict[int, str] = {}
    for loc_id, body in file_locations.items():
        fields = dict(_FIELD_RE.findall(body))
        file_id = int(fields.get("file_name_id", 0))
        line = fields.get("line")
        file_name = file_names.get(file_id, "").strip('"')
        if file_name and line is not None:
            location_source[loc_id] = f"{file_name}:{line}"

    frame_source: dict[int, str] = {}
    for frame_id, body in stack_frames.items():
        fields = dict(_FIELD_RE.findall(body))
        loc_id = int(fields.get("file_location_id", 0))
        if loc_id in location_source:
            frame_source[frame_id] = location_source[loc_id]
    return frame_source


def _parse_hlo_file(path: Path, source_map: JaxKernelSourceMap) -> None:
    """Parse one optimized-HLO text file into ``source_map``."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        console_warning("jax trace", f"Could not read HLO dump {path.name}: {exc}")
        return

    function: Optional[str] = None
    file_names: dict[int, str] = {}
    file_locations: dict[int, str] = {}
    stack_frames: dict[int, str] = {}

    index = 0
    while index < len(lines):
        line = lines[index]
        header = _HLO_MODULE_RE.match(line)
        if header:
            module_name = header.group(1)
            function = (
                module_name[len(_MODULE_NAME_PREFIX) :]
                if module_name.startswith(_MODULE_NAME_PREFIX)
                else module_name
            )
            index += 1
            continue

        section = line.strip()
        if section == "FileNames":
            file_names, index = _parse_indexed_section(lines, index + 1)
            continue
        if section == "FunctionNames":
            _, index = _parse_indexed_section(lines, index + 1)
            continue
        if section == "FileLocations":
            file_locations, index = _parse_indexed_section(lines, index + 1)
            continue
        if section == "StackFrames":
            stack_frames, index = _parse_indexed_section(lines, index + 1)
            continue
        index += 1

    if function is None:
        return

    frame_source = _build_source_table(file_names, file_locations, stack_frames)

    for line in lines:
        instruction = _INSTRUCTION_RE.match(line)
        if not instruction:
            continue
        op_match = _OP_NAME_RE.search(line)
        if not op_match:
            continue
        operator = op_match.group(1)
        frame_match = _STACK_FRAME_RE.search(line)
        source = ""
        if frame_match:
            source = frame_source.get(int(frame_match.group(1)), "")
        shape = _LAYOUT_RE.sub("", instruction.group(2)).strip()
        source_map._add(function, instruction.group(1), operator, source, shape)


def build_jax_kernel_source_map(hlo_dump_dir: Path) -> JaxKernelSourceMap:
    """Build a kernel-to-operator/source map from a workload's HLO dump.

    Returns an empty map when the dump directory is absent or contains no
    optimized-HLO text files.
    """
    source_map = JaxKernelSourceMap()
    if not hlo_dump_dir.is_dir():
        console_debug(f"No HLO dump directory at {hlo_dump_dir}")
        return source_map

    hlo_files = sorted(hlo_dump_dir.glob(_HLO_TEXT_GLOB))
    if not hlo_files:
        console_warning(
            "jax trace",
            f"No optimized-HLO text files found in {hlo_dump_dir}; "
            "operator and source attribution will be unavailable.",
        )
        return source_map

    for path in hlo_files:
        _parse_hlo_file(path, source_map)
    console_debug(
        f"Parsed {len(hlo_files)} HLO dump file(s) from {hlo_dump_dir} "
        f"for {len(source_map._exact)} function(s)."
    )
    return source_map
