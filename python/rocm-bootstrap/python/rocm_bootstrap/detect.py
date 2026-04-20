"""GPU detection orchestration.

Detects AMD GPUs on the current system by reading kernel sysfs interfaces.
All I/O is delegated to the :mod:`rocm_bootstrap._platform` module so that
tests can monkeypatch it with fake sysfs content.

Detection chain:
    1. ``ROCM_BOOTSTRAP_DISABLE_DETECTION`` → return ``[]``
    2. ``ROCM_BOOTSTRAP_FORCE_GFX_ARCH`` → parse forced targets
    3. KFD topology (``/sys/class/kfd/kfd/topology/nodes/*/properties``)
    4. Return empty list if KFD is not available
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from rocm_bootstrap import _platform
from rocm_bootstrap.targets import (
    GfxTarget,
    lookup_target,
    packaging_chain,
    parse_gfx_target_version,
)


@dataclass(frozen=True)
class DetectedGpu:
    """A GPU detected on the system.

    Attributes:
        target: The :class:`~rocm_bootstrap.targets.GfxTarget` for this GPU.
        node_id: KFD topology node index.
        gpu_id: KFD ``gpu_id`` property. 0 for CPU-only nodes.
        pci_id: PCI device ID string if available (e.g., ``"0x7550"``).
    """

    target: GfxTarget
    node_id: int
    gpu_id: int = 0
    pci_id: str | None = None


def detect_gpus() -> list[DetectedGpu]:
    """Detect AMD GPUs on the current system.

    Returns a list of :class:`DetectedGpu` instances, one per GPU found.
    Returns an empty list if detection is disabled, no GPUs are found,
    or the system lacks the required sysfs interfaces.

    Environment variables:
        ``ROCM_BOOTSTRAP_DISABLE_DETECTION``:
            Set to ``1`` to skip detection entirely.
        ``ROCM_BOOTSTRAP_FORCE_GFX_ARCH``:
            Comma-separated list of GFX target names to return instead
            of detecting (e.g., ``"gfx942,gfx942"`` for two MI300X GPUs).
    """
    # 1. Check disable flag
    if _platform.get_env("ROCM_BOOTSTRAP_DISABLE_DETECTION") == "1":
        return []

    # 2. Check forced arch override
    forced = _platform.get_env("ROCM_BOOTSTRAP_FORCE_GFX_ARCH")
    if forced:
        return _parse_forced_targets(forced)

    # 3. KFD topology (only supported detection method)
    return _detect_via_kfd_topology()


def detect_gfx_targets() -> list[GfxTarget]:
    """Convenience: detect GPUs and return deduplicated GfxTarget list.

    Returns unique targets in detection order (first occurrence kept).
    """
    seen: set[str] = set()
    targets: list[GfxTarget] = []
    for gpu in detect_gpus():
        if gpu.target.name not in seen:
            seen.add(gpu.target.name)
            targets.append(gpu.target)
    return targets


# ---------------------------------------------------------------------------
# Internal detection methods
# ---------------------------------------------------------------------------


def _parse_forced_targets(forced: str) -> list[DetectedGpu]:
    """Parse ROCM_BOOTSTRAP_FORCE_GFX_ARCH into DetectedGpu list."""
    gpus: list[DetectedGpu] = []
    for i, name in enumerate(forced.split(",")):
        name = name.strip()
        if not name:
            continue
        target = lookup_target(name)
        gpus.append(DetectedGpu(target=target, node_id=i))
    return gpus


def _parse_kfd_properties(text: str) -> dict[str, int]:
    """Parse KFD properties file into a dict of name → int value.

    Lines are formatted as ``key value`` pairs, one per line. Non-integer
    values are skipped.
    """
    props: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                props[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return props


def _detect_via_kfd_topology() -> list[DetectedGpu]:
    """Detect GPUs via KFD topology sysfs nodes."""
    nodes = _platform.list_kfd_nodes()
    if not nodes:
        return []

    gpus: list[DetectedGpu] = []
    for node_path in nodes:
        try:
            text = _platform.read_kfd_properties(node_path)
        except FileNotFoundError:
            continue

        props = _parse_kfd_properties(text)

        # Skip CPU-only nodes (simd_count == 0 indicates no GPU CUs)
        simd_count = props.get("simd_count", 0)
        if simd_count == 0:
            continue

        gtv = props.get("gfx_target_version", 0)
        if gtv == 0:
            continue

        target = parse_gfx_target_version(gtv)

        node_id = int(node_path.name)
        gpu_id = props.get("gpu_id", 0)
        device_id = props.get("device_id")
        pci_id = f"0x{device_id:x}" if device_id is not None else None

        gpus.append(
            DetectedGpu(
                target=target,
                node_id=node_id,
                gpu_id=gpu_id,
                pci_id=pci_id,
            )
        )

    return gpus


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``rocm-bootstrap-detect``."""
    parser = argparse.ArgumentParser(
        prog="rocm-bootstrap-detect",
        description="Detect AMD GPUs and print target information.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--unique",
        "-u",
        action="store_const",
        dest="mode",
        const="unique",
        help="Print unique target names, one per line.",
    )
    group.add_argument(
        "--verbose",
        "-v",
        action="store_const",
        dest="mode",
        const="verbose",
        help="Human-readable output with hierarchy and generic ISA details.",
    )
    group.add_argument(
        "--hierarchy",
        action="store_const",
        dest="mode",
        const="hierarchy",
        help="Print unique bundle hierarchies (target sub-family family).",
    )
    parser.set_defaults(mode="unique")
    args = parser.parse_args(argv)

    gpus = detect_gpus()

    if args.mode == "verbose":
        _print_verbose(gpus)
    elif args.mode == "hierarchy":
        _print_hierarchy(gpus)
    else:
        _print_unique(gpus)


def _print_unique(gpus: list[DetectedGpu]) -> None:
    """Machine-consumable: one unique target name per line."""
    for target in _unique_targets(gpus):
        print(target.name)


def _print_hierarchy(gpus: list[DetectedGpu]) -> None:
    """Unique packaging chains, space-separated: target sub_family family."""
    seen: set[str] = set()
    for target in _unique_targets(gpus):
        if target.name in seen:
            continue
        seen.add(target.name)
        chain = packaging_chain(target)
        print(" ".join(b.key for b in chain))


def _print_verbose(gpus: list[DetectedGpu]) -> None:
    """Human-readable output with full details."""
    if not gpus:
        print("No AMD GPUs detected.")
        return

    print(f"Detected {len(gpus)} AMD GPU(s):\n")
    for gpu in gpus:
        t = gpu.target
        parts = [f"  Node {gpu.node_id}: {t.name}"]
        parts.append(f"major={t.major} minor={t.minor} stepping={t.stepping}")
        if gpu.pci_id:
            parts.append(f"PCI={gpu.pci_id}")
        if gpu.gpu_id:
            parts.append(f"gpu_id={gpu.gpu_id}")
        print("  ".join(parts))

        chain = packaging_chain(t)
        # chain is (target_bundle, sub_family_bundle, family_bundle)
        for bundle in chain[1:]:  # skip target-level (already printed)
            label = bundle.level.value.replace("_", "-")
            line = f"    {label}: {bundle.key} ({bundle.display_name})"
            if bundle.llvm_generic:
                line += f"  generic: {bundle.llvm_generic}"
            print(line)
        print()


def _unique_targets(gpus: list[DetectedGpu]) -> list[GfxTarget]:
    """Deduplicate targets preserving detection order."""
    seen: set[str] = set()
    targets: list[GfxTarget] = []
    for gpu in gpus:
        if gpu.target.name not in seen:
            seen.add(gpu.target.name)
            targets.append(gpu.target)
    return targets


if __name__ == "__main__":
    main()
