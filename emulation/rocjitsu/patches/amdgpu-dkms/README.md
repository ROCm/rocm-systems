# amdgpu-dkms patches for rocjitsu vfio-user

These patches apply to the `amdgpu-dkms` package installed from the ROCm 6.4
repository. They are required to make `modprobe amdgpu` succeed when the GPU
is a rocjitsu vfio-user emulated device rather than real hardware.

## Applying

```bash
# Inside the guest VM (adjust version as needed):
AMDGPU_SRC="/usr/src/amdgpu-6.19.4-2337710.24.04"
AMDGPU_VER="6.19.4-2337710.24.04"
PATCH_DIR="/home/stebates/Projects/rocm-systems/emulation/rocjitsu/patches/amdgpu-dkms"

# Apply patches to source tree
sudo patch -p1 --forward -d "$AMDGPU_SRC" < "$PATCH_DIR/0001-vfu-invalid-rreg-no-bug.patch"
sudo patch -p1 --forward -d "$AMDGPU_SRC" < "$PATCH_DIR/0002-vfu-mm-access-skip-drm-dev-enter.patch"

# Use the helper script for incremental rebuild and reload
"$PATCH_DIR/rebuild-and-reload.sh"
```

## Patches

### 0001-vfu-invalid-rreg-no-bug.patch

**Problem:** During `amdgpu_discovery_set_ip_blocks()`, the driver calls
`RREG32(mmMP0_SMN_C2PMSG_33)` (register 0x16061, byte offset 0x58184).
This address exceeds the 256 KB direct BAR5 window, so `amdgpu_device_rreg()`
routes it through `adev->pcie_rreg`, which is not yet initialized at probe
time. This calls `amdgpu_invalid_rreg()` which hits `BUG()` and crashes the
kernel.

**Fix:** Replace `BUG()` with `pr_warn_once()` + `return 0`. The C2PMSG_33
polling loop exhausts its 1000 iterations returning 0, then the driver reads
`mmRCC_CONFIG_MEMSIZE` (offset 0x378C, within BAR5) which our BAR5 model
returns as 256 (MB). The driver then reads the IP discovery binary from VRAM
at `256 MB - DISCOVERY_TMR_OFFSET`, which rocjitsu writes at startup.

Also adds `-Wno-unused-variable` to `amd/amdgpu/Makefile` to suppress a
pre-existing warning (`dev_attr_pcie_replay_count`) that causes the DKMS
build to fail under `-Werror`.

**Applies to:** amdgpu-dkms 6.12.12-2147987.24.04 (ROCm 6.4 / Ubuntu 24.04)

**Note:** This is not appropriate for upstream submission. The correct
long-term fix is to initialize `pcie_rreg` before IP discovery runs, or
to stub C2PMSG_33 via the NBIO IP block initialization sequence.
