# VFIO pod guest payloads and external inputs

These guest files support the two-GPU peer refetch probe and the four-node,
eight-rank RCCL pod gate. Build the host server with `ROCJITSU_ENABLE_VFIO=ON`
as described in [the QEMU guide](../../docs/qemu-vfio.md). The guest kernel,
AMDGPU DKMS package, firmware, ROCm user libraries, and BusyBox image are
external inputs; use mutually compatible versions and record their SHA-256
hashes with every result.

## Pinned transport and driver inputs

The reported 2026-10-15 and 2026-10-19 **final** VFIO pod runs used:

| Input | Version or source | Purpose |
| --- | --- | --- |
| QEMU | 10.1.0, commit `f8b2f64e2336a28bf0d50b6ef8a7d8c013e9bcf3` | Two `vfio-user-pci` functions and one NIC per guest |
| libvfio-user | commit `37491ed9af828fc161238dacd82e83ea35a09f87` | Standard VFIO-user DMA mapping; CMake fetches this revision |
| AMDGPU DKMS | `amdgpu-dkms_7.1.3.31500000-2390945.24.04_all.deb` | Guest `amdgpu.ko` and companion modules |
| Linux image and headers | `6.17.0-23-generic` | Guest `vmlinuz` and matching module build tree |
| ROCm guest user stack | HIP `7.14.60850-0000000`, ROCr `1.21`, RCCL `2.30.4-HEAD:2b22ab0` | Rank binary and runtime libraries |

Apply the three patches in `patches/qemu/10.1/` with
`scripts/apply-qemu-pcie-fixes.sh` to a **clean** checkout at the pinned QEMU
commit. They preserve the VFIO-user class code, supply a valid XIO3130
downstream LinkCap, and advertise opt-in root/switch AtomicOp support for
implemented guest-RAM atomics. They add no private VFIO-user protocol. For
example, from the rocJITsu source directory:

```sh
git clone https://gitlab.com/qemu-project/qemu.git /path/to/qemu-10.1-src
git -C /path/to/qemu-10.1-src checkout f8b2f64e2336a28bf0d50b6ef8a7d8c013e9bcf3
scripts/apply-qemu-pcie-fixes.sh /path/to/qemu-10.1-src
mkdir -p /path/to/qemu-10.1-build
cd /path/to/qemu-10.1-build
/path/to/qemu-10.1-src/configure --target-list=x86_64-softmmu --enable-kvm --disable-werror
ninja qemu-system-x86_64
```

Use the produced binary's `-device help` output to confirm `vfio-user-pci`.
The rocJITsu VFIO CMake build fetches libvfio-user at the revision above and
applies `patches/libvfio-user/0001-preserve-small-64-bit-bar-upper-address.patch`
automatically. That general BAR mask fix preserves the upper half of a
sub-4-GiB 64-bit peer BAR. Neither the QEMU nor libvfio-user patch adds a
rocJITsu-specific peer-map command.

For the **zero-patch** final gate, extract the pinned DKMS package and build
against headers for the *guest* kernel. Its `Makefile` handles configuration
and module dependencies:

```sh
dpkg-deb -x /path/to/amdgpu-dkms_7.1.3.31500000-2390945.24.04_all.deb /path/to/driver-tree
make -C /path/to/driver-tree/usr/src/amdgpu-7.1.3-2390945.24.04 \
  KERNELVER=6.17.0-23-generic
modinfo -F vermagic /path/to/driver-tree/usr/src/amdgpu-7.1.3-2390945.24.04/amd/amdgpu/amdgpu.ko
```

The build host needs `/lib/modules/6.17.0-23-generic/build` with the matching
kernel `.config` and `Module.symvers`. Copy the resulting `amdgpu.ko` and its
companion modules into the guest image in dependency order. Use the matching
`/boot/vmlinuz-6.17.0-23-generic` as the launcher's `--kernel`. Record
`sha256sum` for the package, module, kernel, QEMU binary, and final image.
The recorded gate used the distribution kernel image; it did not apply a
rocJITsu kernel patch or build a custom kernel. If rebuilding that kernel,
start from the source package for the same release with its exact config and
symbol versions, then rebuild the DKMS module against the resulting headers
and record the new binary hashes. A kernel with a matching version string but
different symbols is not an interchangeable module build target.

The October 14 diagnostic run used local AMDGPU error-path fixes, but the
October 15 and 19 final runs used **none** of those driver changes. The
diagnostic patches are not included in this repository or needed by the pod
controller, which checks directed fabric effects and guest workload logs.
Do not report a diagnostic module as the zero-patch result.

## Build and pack the guest payloads

Build the peer refetch probe with a gfx1250-capable `hipcc`. The same guest
image must include its loader and the transitive HIP/ROCr libraries:

```sh
hipcc --offload-arch=gfx1250 -O2 \
  -o /path/to/hip_owner_peer_refetch tests/guest/hip_owner_peer_refetch.cpp
cmake --build build-vfio --target hip_rccl_gfx1250_test_target
```

Start from a compatible guest staging root containing BusyBox, the matching
kernel modules and firmware, network applets, ROCm libraries, and their ELF
dependencies. `docs/qemu-vfio.md` explains the firmware inventory and
`rj-ip-discovery` output. The pod init reads `/modules/load-order`, expects
ROCm under `/opt/rocm/core-7.14/`, and runs the rank binary at the path below.
The peer refetch init uses the same module list and an independent probe
binary. Pack separate images for the two workloads:

```sh
mkdir -p /path/to/pod-root /path/to/refetch-root
cp -a /path/to/compatible-base-root/. /path/to/pod-root/
install -D -m 0755 tests/guest/rccl_pod_init /path/to/pod-root/init
install -D -m 0755 build-vfio/tests/hip_rccl_gfx1250_test \
  /path/to/pod-root/opt/rocjitsu/bin/hip_rccl_gfx1250_test
(cd /path/to/pod-root && find . -print0 | cpio --null -o -H newc) \
  | gzip -9 > /path/to/pod-initramfs.gz

cp -a /path/to/compatible-base-root/. /path/to/refetch-root/
install -D -m 0755 tests/guest/hip_owner_peer_refetch_init /path/to/refetch-root/init
install -D -m 0755 /path/to/hip_owner_peer_refetch \
  /path/to/refetch-root/opt/rocjitsu/bin/hip_owner_peer_refetch
(cd /path/to/refetch-root && find . -print0 | cpio --null -o -H newc) \
  | gzip -9 > /path/to/refetch-initramfs.gz
```

Run the pod image with `scripts/run-rccl-pod-vfio.py`, the four-node manifest
`configs/gfx1250_mi455x_kmd_4node_2gpu_if.json`, and the exact external
artifact paths. The controller checks the guest's printed module hash against
`--driver-module`; the module staged in the image must be the same build.

```sh
python3 scripts/run-rccl-pod-vfio.py \
  --qemu /path/to/qemu-10.1-build/qemu-system-x86_64 \
  --kernel /boot/vmlinuz-6.17.0-23-generic \
  --initramfs /path/to/pod-initramfs.gz \
  --rocjitsu build-vfio/tools/rocjitsu/rocjitsu \
  --config configs/gfx1250_mi455x_kmd_4node_2gpu_if.json \
  --driver-module /path/to/driver-tree/usr/src/amdgpu-7.1.3-2390945.24.04/amd/amdgpu/amdgpu.ko \
  --driver-package /path/to/amdgpu-dkms_7.1.3.31500000-2390945.24.04_all.deb \
  --rank-binary build-vfio/tests/hip_rccl_gfx1250_test \
  --output /tmp/rocjitsu-pod-vfio --timeout 600
```

The dated qualification record gives the original commands and binary
hashes. For the peer probe, run one two-GPU node with
`scripts/run-vfio-guest.py`:

```sh
python3 scripts/run-vfio-guest.py \
  --qemu /path/to/qemu-10.1-build/qemu-system-x86_64 \
  --kernel /boot/vmlinuz-6.17.0-23-generic \
  --initramfs /path/to/refetch-initramfs.gz \
  --rocjitsu build-vfio/tools/rocjitsu/rocjitsu \
  --config configs/gfx1250_mi455x_kmd_4node_2gpu_if.json \
  --node node0 --socket /tmp/refetch-vfio/gpu0.sock \
  --socket /tmp/refetch-vfio/gpu1.sock \
  --output /tmp/refetch-vfio --accel tcg --memory 12G \
  --append 'console=ttyS0 rdinit=/init panic=-1 amdgpu.discovery=2 amdgpu.emu_mode=1 amdgpu.fw_load_type=0 amdgpu.vm_update_mode=3 amdgpu.gpu_recovery=0 amdgpu.vramlimit=1024 amdgpu.ip_block_mask=0x3f' \
  --expect-log 'guest: owner-peer-refetch PASS' \
  --reject-log 'guest: owner-peer-refetch FAIL'
```

Use a fresh, short `--output` path. The owner probe uses three ordinary
`global_load_b32` instructions without a system scope cache modifier and no
owner-side invalidate. The separate ready/done pages use system scope traffic
for ordering. The probe reports the first cached owner value, its repeated
hit, and the post-peer-write refetch; `result=PASS` is required. The
`TranslatedLocalCache.ExactHandlePtePolicyAndFabricBoundaryRefetchOwnerData`
unit test also establishes the negative control: after an external backing
write, a V$ invalidation alone still returns the stale L2 value, while full
completion invalidation refetches the new value. These tests exercise UC PCIe
peer mappings through rocJITsu's conservative fabric coherence model. They do
not establish native xGMI or guest cache-maintenance ordering.
