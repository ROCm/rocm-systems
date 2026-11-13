# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Normalize physical GPU metadata shared by ROCr, AMD-SMI, and the SDK."""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, TypeVar

DeviceHandle = TypeVar("DeviceHandle")
_PHYSICAL_FIELDS = ("node_id", "bdf_id", "uuid")
_MISSING_TEXT = {"", "none", "nan", "n/a", "na", "unknown", "not supported"}
_INTEGER_SENTINELS = {-1, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF}


def identity_from_specs(metadata: Mapping[str, Any]) -> "GpuIdentity":
    """Read the stable identity fields in live or saved machine specifications."""
    return GpuIdentity(
        arch=metadata.get("gpu_arch"),
        chip_id=metadata.get("gpu_chip_id"),
        node_id=metadata.get("gpu_node_id"),
        bdf_id=metadata.get("gpu_bdf_id"),
        uuid=metadata.get("gpu_uuid"),
    )


def match_gpu_identity(
    selected: "GpuIdentity",
    candidates: Sequence[Tuple[DeviceHandle, "GpuIdentity"]],
    source: str,
) -> DeviceHandle:
    """Require a unique physical match with consistent overlapping metadata.

    A missing stronger field permits BDF/UUID fallback. A conflicting stronger
    field excludes that candidate; a duplicate stronger match is ambiguous.
    Metadata-free discovery can use a sole device, preserving older single-GPU
    systems whose ROCr and AMD-SMI versions expose no physical identifiers.
    """
    remaining = list(candidates)
    for identity_field in _PHYSICAL_FIELDS:
        selected_value = getattr(selected, identity_field)
        if selected_value is None:
            continue
        matches = [
            (handle, identity)
            for handle, identity in remaining
            if getattr(identity, identity_field) == selected_value
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Cannot match the selected GPU to exactly one {source} device: "
                f"ambiguous {identity_field}={selected_value!r}."
            )
        if matches:
            handle, identity = matches[0]
            _validate_identity_match(selected, identity, source)
            return handle
        remaining = [
            (handle, identity)
            for handle, identity in remaining
            if getattr(identity, identity_field) is None
        ]
        if not remaining:
            raise ValueError(
                f"Cannot match the selected GPU to exactly one {source} device: "
                f"conflicting or missing {identity_field}={selected_value!r}."
            )
    if not selected.has_physical_identity and len(remaining) == 1:
        handle, identity = remaining[0]
        _validate_identity_match(selected, identity, source)
        return handle
    raise ValueError(f"The selected GPU lacks a unique {source} physical identity.")


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _MISSING_TEXT else text


def _normalize_integer(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        return None
    text = _normalize_text(value)
    if text is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return None if number < 0 or number in _INTEGER_SENTINELS else number


def _normalize_uuid(value: object) -> Optional[str]:
    text = _normalize_text(value)
    if text is None:
        return None
    match = re.fullmatch(r"GPU-([0-9a-f]+)", text, flags=re.IGNORECASE)
    if match is None or set(match.group(1).lower()) <= {"0"}:
        return None
    return f"GPU-{match.group(1).lower()}"


def _validate_identity_match(
    selected: "GpuIdentity", candidate: "GpuIdentity", source: str
) -> None:
    for identity_field in ("arch", "chip_id") + _PHYSICAL_FIELDS:
        selected_value = getattr(selected, identity_field)
        candidate_value = getattr(candidate, identity_field)
        if (
            selected_value is not None
            and candidate_value is not None
            and selected_value != candidate_value
        ):
            raise ValueError(
                f"Cannot match the selected GPU to exactly one {source} device: "
                f"conflicting {identity_field} metadata "
                f"({selected_value!r} != {candidate_value!r})."
            )


@dataclass(frozen=True)
class GpuIdentity:
    """Stable GPU identity, independent of discovery order and adapter handles."""

    arch: Optional[str] = None
    chip_id: Optional[str] = None
    node_id: Optional[int] = None
    bdf_id: Optional[int] = None
    uuid: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arch", _normalize_text(self.arch))
        chip_number = _normalize_integer(self.chip_id)
        object.__setattr__(
            self, "chip_id", str(chip_number) if chip_number is not None else None
        )
        object.__setattr__(self, "node_id", _normalize_integer(self.node_id))
        bdf_number = _normalize_integer(self.bdf_id)
        # ROCr/SDK location IDs omit the PCI domain in AMD-SMI's high 32 bits.
        object.__setattr__(
            self, "bdf_id", bdf_number & 0xFFFFFFFF if bdf_number is not None else None
        )
        object.__setattr__(self, "uuid", _normalize_uuid(self.uuid))

    @property
    def has_physical_identity(self) -> bool:
        return any(getattr(self, name) is not None for name in _PHYSICAL_FIELDS)

    def spec_fields(self) -> Dict[str, Any]:
        """Return normalized physical metadata using machine-spec field names."""
        return {
            f"gpu_{name}": getattr(self, name)
            for name in _PHYSICAL_FIELDS
            if getattr(self, name) is not None
        }
