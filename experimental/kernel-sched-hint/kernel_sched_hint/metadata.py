#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
###############################################################################
"""Kernel record the runtime already has after COMGR parsing.

``device::Kernel::GetAttrCodePropMetadata`` and ``InitParameters`` in
``projects/clr/rocclr/device/devkernel.cpp`` walk this metadata.
``roc::Kernel::init`` calls that, and ``VirtualGPU::submitKernelInternal``
in ``rocvirtual.cpp`` then has the signature, VGPR/SGPR counts, group and
private segment sizes, and the launch (grid plus ``sharedMemBytes``).

The field names match code-object v3 keys in ``platform/kernel_init.hpp``
(``.vgpr_count``, ``.sgpr_count``, ``.args`` / ``.value_kind``, ...).
``amd_comgr_lookup_code_object`` reports ``amd_comgr_code_object_info_t::size``
for the whole ISA blob; that is ``code_object_size`` here, not one kernel.
Per-kernel machine-code size is the ELF symbol size when the caller has it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Argument:
    """One ``.args`` entry. Hidden kinds are the runtime's, not the user's."""

    value_kind: str
    size: int = 0
    address_space: str = ""
    access: str = ""
    name: str = ""
    hidden: bool = False


@dataclass
class KernelMetadata:
    name: str = ""
    vgprs: int = 0
    sgprs: int = 0
    vgpr_spill_count: int = 0
    sgpr_spill_count: int = 0
    wavefront_size: int = 0
    group_segment_fixed_size: int = 0
    private_segment_fixed_size: int = 0
    kernarg_segment_size: int = 0
    max_flat_workgroup_size: int = 0
    # Machine-code bytes of this kernel. 0 if the caller only has the blob.
    isa_size: int = 0
    # amd_comgr_code_object_info_t::size for the ISA in the fat binary.
    code_object_size: int = 0
    kind: str = ""
    arguments: list[Argument] = field(default_factory=list)
    # Launch-time. submitKernelInternal adds sharedMemBytes to group segment.
    dynamic_shared_bytes: int = 0
    # Allocation sizes for user global_buffer args, in argument order.
    # processMemObjects already has mem->getSize() for each pointer.
    global_buffer_bytes: list[int] = field(default_factory=list)

    def user_arguments(self) -> list[Argument]:
        return [
            arg
            for arg in self.arguments
            if not arg.hidden and not _is_hidden_kind(arg.value_kind)
        ]

    def global_buffers(self) -> list[Argument]:
        return [
            arg
            for arg in self.user_arguments()
            if arg.value_kind in ("global_buffer", "GlobalBuffer")
        ]


def _is_hidden_kind(kind: str) -> bool:
    return kind.startswith("hidden_") or kind.startswith("Hidden")


_KERNEL_INT_FIELDS = {
    ".vgpr_count": "vgprs",
    ".sgpr_count": "sgprs",
    ".vgpr_spill_count": "vgpr_spill_count",
    ".sgpr_spill_count": "sgpr_spill_count",
    ".wavefront_size": "wavefront_size",
    ".group_segment_fixed_size": "group_segment_fixed_size",
    ".private_segment_fixed_size": "private_segment_fixed_size",
    ".kernarg_segment_size": "kernarg_segment_size",
    ".max_flat_workgroup_size": "max_flat_workgroup_size",
    ".isa_size": "isa_size",
    # Code-object v2 names, from kCodePropFieldMap.
    "NumVGPRs": "vgprs",
    "NumSGPRs": "sgprs",
    "NumSpilledVGPRs": "vgpr_spill_count",
    "NumSpilledSGPRs": "sgpr_spill_count",
    "WavefrontSize": "wavefront_size",
    "GroupSegmentFixedSize": "group_segment_fixed_size",
    "PrivateSegmentFixedSize": "private_segment_fixed_size",
    "KernargSegmentSize": "kernarg_segment_size",
    "MaxFlatWorkGroupSize": "max_flat_workgroup_size",
}

_ARG_INT_FIELDS = {
    ".size": "size",
    "Size": "size",
}


def parse_amdgpu_metadata(text: str) -> list[KernelMetadata]:
    """Parse a code-object metadata block or the ``.amdhsa_*`` directives.

    Accepts the YAML subset LLVM prints between ``.amdgpu_metadata`` and
    ``.end_amdgpu_metadata``. A kernel with only directives and no YAML
    still yields VGPR/SGPR counts when those directives are present.
    """

    body = text
    match = re.search(
        r"\.amdgpu_metadata\b(.*?)(?:\.end_amdgpu_metadata|\Z)",
        text,
        re.S,
    )
    kernels: list[KernelMetadata] = []
    if match:
        kernels = _parse_yaml_block(match.group(1))
        body = text

    if not kernels:
        kernels = _parse_directives(body)
    else:
        _overlay_directives(kernels, body)
    return kernels


def _parse_directives(text: str) -> list[KernelMetadata]:
    names = re.findall(r"\.amdhsa_kernel\s+(\S+)", text)
    if not names:
        return []
    found = []
    for name in names:
        meta = KernelMetadata(name=name)
        _fill_directives(meta, text, name)
        found.append(meta)
    return found


def _overlay_directives(kernels: list[KernelMetadata], text: str) -> None:
    by_name = {kernel.name: kernel for kernel in kernels}
    for name in re.findall(r"\.amdhsa_kernel\s+(\S+)", text):
        meta = by_name.get(name)
        if meta is None:
            meta = KernelMetadata(name=name)
            kernels.append(meta)
            by_name[name] = meta
        _fill_directives(meta, text, name)


def _fill_directives(meta: KernelMetadata, text: str, name: str) -> None:
    window = text
    match = re.search(
        rf"\.amdhsa_kernel\s+{re.escape(name)}\b(.*?)(?:\.end_amdhsa_kernel|\.amdhsa_kernel|\Z)",
        text,
        re.S,
    )
    if match:
        window = match.group(1)
    vg = re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", window)
    sg = re.search(r"\.amdhsa_next_free_sgpr\s+(\d+)", window)
    if vg and meta.vgprs <= 0:
        meta.vgprs = int(vg.group(1))
    if sg and meta.sgprs <= 0:
        meta.sgprs = int(sg.group(1))


def _parse_yaml_block(body: str) -> list[KernelMetadata]:
    kernels: list[KernelMetadata] = []
    current: KernelMetadata | None = None
    current_arg: Argument | None = None
    in_args = False
    args_indent = -1

    for raw in body.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if line == "---" or line.startswith("amdhsa."):
            if line.startswith("amdhsa.kernels"):
                in_args = False
            continue

        if line.startswith("- "):
            item = line[2:].strip()
            if in_args and indent >= args_indent:
                current_arg = Argument(value_kind="")
                if current is not None:
                    current.arguments.append(current_arg)
                _assign_arg_field(current_arg, item)
                continue
            in_args = False
            current_arg = None
            if item.startswith(".name:") or item.startswith("Name:"):
                current = KernelMetadata(name=_scalar(item))
                kernels.append(current)
            continue

        if current is None:
            continue
        key = line.split(":", 1)[0].strip()
        if key in (".args", "Args"):
            in_args = True
            args_indent = indent + 1
            current_arg = None
            continue
        if in_args and current_arg is not None and indent >= args_indent:
            _assign_arg_field(current_arg, line)
            continue
        if in_args and indent < args_indent:
            in_args = False
            current_arg = None
        _assign_kernel_field(current, line)
    for kernel in kernels:
        for arg in kernel.arguments:
            if _is_hidden_kind(arg.value_kind):
                arg.hidden = True
    return kernels


def _scalar(item: str) -> str:
    _, _, value = item.partition(":")
    return value.strip().strip("'\"")


def _assign_kernel_field(kernel: KernelMetadata, line: str) -> None:
    key, _, value = line.partition(":")
    key = key.strip()
    value = value.strip().strip("'\"")
    if not value:
        return
    attr = _KERNEL_INT_FIELDS.get(key)
    if attr is not None and re.fullmatch(r"-?\d+", value):
        setattr(kernel, attr, int(value))
        return
    if key in (".name", "Name", "SymbolName", ".symbol") and not kernel.name:
        kernel.name = value
    elif key in (".kind", "Kind"):
        kernel.kind = value


def _assign_arg_field(arg: Argument, line: str) -> None:
    key, _, value = line.partition(":")
    key = key.strip()
    value = value.strip().strip("'\"")
    if not value:
        return
    if key in (".value_kind", "ValueKind"):
        arg.value_kind = value
        arg.hidden = _is_hidden_kind(value)
    elif key in (".address_space", "AddrSpaceQual"):
        arg.address_space = value
    elif key in (".access", ".actual_access", "AccQual", "ActualAccQual"):
        if key in (".actual_access", "ActualAccQual") or not arg.access:
            arg.access = value
    elif key in (".name", "Name"):
        arg.name = value
    elif key in _ARG_INT_FIELDS and re.fullmatch(r"-?\d+", value):
        setattr(arg, _ARG_INT_FIELDS[key], int(value))
