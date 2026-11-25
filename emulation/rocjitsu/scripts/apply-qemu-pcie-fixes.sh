#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/apply-qemu-pcie-fixes.sh QEMU_SOURCE

Apply three general PCIe fixes to a clean QEMU 10.1.0 checkout at
f8b2f64e2336a28bf0d50b6ef8a7d8c013e9bcf3. No VFIO-user protocol
extension or device-specific peer-map property is added.

The VFIO-user class-code fix preserves the class read from the device.
The XIO3130 link patch supplies a valid Gen1 x1 LinkCap on each downstream
port; QEMU 10.1 otherwise overwrites that capability with zero link fields.

The PCIe capability patch adds explicit root-completer and switch-routing
properties, disabled by default. Enable them only when the DMA backend
provides the corresponding operations.
EOF
}

if (( $# != 1 )); then
    usage
    exit 2
fi

source_dir=$1

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
patch_dir=$script_dir/../patches/qemu/10.1
expected_head=f8b2f64e2336a28bf0d50b6ef8a7d8c013e9bcf3
actual_head=$(git -C "$source_dir" rev-parse HEAD)
if [[ $actual_head != "$expected_head" ]]; then
    echo "QEMU source is at $actual_head, expected $expected_head" >&2
    exit 1
fi
if [[ -n $(git -C "$source_dir" status --porcelain --untracked-files=no) ]]; then
    echo "QEMU source has tracked edits; use a clean checkout" >&2
    exit 1
fi

class_patch=$patch_dir/qemu-10.1-vfio-user-class-code-default.patch
link_patch=$patch_dir/qemu-10.1-xio3130-link-capability.patch
atomic_patch=$patch_dir/qemu-10.1-pcie-atomic-capabilities.patch
git -C "$source_dir" apply --check "$class_patch"
git -C "$source_dir" apply --check --unidiff-zero "$atomic_patch"
git -C "$source_dir" apply --check --unidiff-zero "$link_patch"

git -C "$source_dir" apply "$class_patch"
git -C "$source_dir" apply --unidiff-zero "$atomic_patch"
git -C "$source_dir" apply --unidiff-zero "$link_patch"
git -C "$source_dir" diff --check
echo "Preserved VFIO-user PCI class code from the device"
echo "Added opt-in PCIe root AtomicOp and switch-routing capabilities"
echo "Applied XIO3130 downstream PCIe link capability fix"
