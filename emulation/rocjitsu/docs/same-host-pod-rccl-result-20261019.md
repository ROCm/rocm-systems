# Same-host pod RCCL local result, 2026-10-19

The saved QEMU/VFIO and emulated KFD runs report the workload outcomes in the
[same-host gate](same-host-pod-rccl-gate.md). Four independent nodes each
hosted two gfx1250 A0 GPUs and one RCCL rank per GPU. Local peer effects
crossed the modeled node-local fabric; inter-node `NET/Socket` traffic used
guest `virtio-net-pci` NICs in QEMU and host loopback in emulated KFD mode.
Every GPU retained separate VRAM.

The missing build-time source state prevents this record from qualifying the
current rebased tree.

Both controllers requested bounded directed effect capture through the
`--fabric-effect-trace` rocJITsu launch option. This diagnostic uses no new
simulator environment variable. The QEMU guests loaded the pinned AMDGPU/
AMDKFD module with zero local patches and used standard VFIO-user DMA maps.
Their stock driver exposed UC PCIe peer BAR mappings, not native xGMI. The
emulated run used ROCr's public `HSA_ENABLE_IPC_MODE_LEGACY=1` setting; its
default IPC mode remains outside this qualification.

The following describes the current VFIO implementation. The saved run did
not capture enough source state to establish that it used this same aperture
path. The VFIO model receives the driver's per-VMID shader aperture settings
through
GC 12.1 `GRBM_GFX_CNTL`, `SH_MEM_CONFIG`, and `SH_MEM_BASES` register writes.
Waves and FLAT shared-memory routing use the selected aperture bank. No
aperture value is seeded by the VFIO server.

The local review output remains under `tmp/pod-review-20261019/evidence/`
in the `rj-review-qemu-retry-20261019` and
`rj-review-emu-retry-20261019` directories;
the earlier no-environment-variable runs remain under
`tmp/pod-noenv-20261019/evidence/`.
These ignored `tmp/` paths are not a durable published archive. Each
`result.json` contains the exact generated node commands, selected
environment, per-rank results, channel records, route totals, socket checks,
and SHA-256 hashes of captured logs.

| Mode | Result JSON SHA-256 | Duration | Node exits | Sockets left |
| --- | --- | ---: | --- | ---: |
| QEMU/VFIO | `44a6e590cbc0dc2e07ba3c9c053df9ebf4653f25979b8c69e81f70892ca41a93` | 95.31 s | `0,0,0,0` | 0 VFIO |
| Emulated KFD | `0b39807b67baa6de9ba45f28b1dc38306525d5f3719ffff1897abc718d775d07` | 56.79 s | `0,0,0,0` | 0 daemon |

## Source and userspace provenance

The preserved rocJITsu executable has the recorded SHA-256
`f7807db20a0a8b5b3dd522e0e185c57d3e1d7c56e74bd617b5273faaad467034`.
Its `--version` output embeds revision
`650e77ceb6144dc005293239dc356dc6f2b5c04c`, whose parent is
`a4c509d4957569cb0747245d993cbea66edb009f`. That identifies the
reported build revision, but the run did not save `git status`, an uncommitted
patch, or a durable source and evidence archive. The saved emulated KFD
`result.json` does contain hashes for its controller and node helper, listed
below. Those hashes identify the historical inputs but do not archive their
bytes. Without saved source state, neither they nor the binary hash prove a
clean checkout or establish that later rebased IPC and shader-aperture fixes
were included.

The emulated KFD rank logs report RCCL `2.30.4-HEAD:2b22ab0`, HIP
`7.14.60850-0000000`, and ROCm `7.14.0.0-9999-2b22ab01` as RCCL build
metadata. RCCL constructs the latter two strings from `HIP_BUILD_INFO` and
`ROCM_BUILD_INFO`; they do not identify the HIP or ROCr libraries loaded for
this run. All 16 saved emulated rank logs report
`Librccl path : /opt/rocm/lib/librccl.so.1`. The rank binary has
`/opt/rocm/lib` in its RUNPATH. The following host files currently resolve
under `/opt/rocm/core-7.14/lib`; their July 9 modification times predate the
run, but no loaded-library hashes or loader map were captured during it.
The reported RCCL path and current hashes cannot prove the executed bytes.
This host library table applies to emulated KFD mode; the VFIO guest used a
separately staged runtime.

| Current emulated-host library file | Current SHA-256 |
| --- | --- |
| `librccl.so.1.0` | `7cd64de6a62b52ed9a457b6d6b9158da49c35fcab44d036f028e84cfa56141a8` |
| `libamdhip64.so.7.14.60850-0000000` | `69e34bce2d4053a5be81c055135a45398a5ca2dcfc0cfec82a5e6901a8f7fb15` |
| `libhsa-runtime64.so.1.21.0` | `15f8c717f4ebf0d00e64b95090466f35fcda78e0a4728fac010af55e91da3395` |

## Inputs and commands

| Input | SHA-256 |
| --- | --- |
| System manifest `configs/gfx1250_mi455x_kmd_4node_2gpu_if.json` | `c14b21b6df774e9fddcba59c93f0c27f5063dcd086d39f38b360bdb0994fcd22` |
| QEMU binary | `3aab60dfb33fe2dac93e89b758f480f9758dfc633b8854363276854d9e132c23` |
| rocJITsu binary | `f7807db20a0a8b5b3dd522e0e185c57d3e1d7c56e74bd617b5273faaad467034` |
| RCCL rank binary | `e40759d0783758a77bdd96317a6825754c7656f4b35c199c04e18f66c2488c7a` |
| Emulated KFD controller `scripts/run-rccl-pod-emulated.py` | `ea831d15bbccb1fc91c12bdda72f2eb0dcfa48d18ac4aec196a6082bf4b9e838` |
| Emulated KFD node helper `scripts/run-rccl-pod-emulated-node.py` | `ec8c10575ca8b5b64e840e4dba80f8cc1c14fe397f6a1a22a68696340b47ca49` |
| Zero-patch guest `amdgpu.ko` | `ffe029d3f994de1d8b5c0ce044854513dd6d31969f562b9fb4e2f374d41c7d13` |
| Pinned AMDGPU driver package | `edf6b629f2cb2d6091aac531c0787aee2387bdc0748a91f21ea38c6c10970ffb` |
| Guest kernel | `e484d1fc35c787c76061191aecb146a851cf9cab37259665f61c9863f93a6194` |
| Guest initramfs | `0056f623c9eadc42065549114a4b1f5c539f3124dac18449bd67fa15fc5003ce` |

The saved node helper differs from this checkout's
`scripts/run-rccl-pod-emulated-node.py` (SHA-256
`942f292de1fe055a578b9006f174aed04cd6a9cf55b397e414d8cec2a41e7734`).
The controller still matches its saved hash. Running the command below with
the current helper does not reproduce the exact recorded script inputs.

```sh
python3 scripts/run-rccl-pod-vfio.py \
  --qemu tmp/p2v-qemu-stock-transport-build/qemu-system-x86_64 \
  --kernel tmp/p3v-guest-coherent-trial/vmlinuz \
  --initramfs tmp/unpatched-kmd-20261015/pod-initramfs.gz \
  --rocjitsu tmp/p3v-alias-observation-release/tools/rocjitsu/rocjitsu \
  --config configs/gfx1250_mi455x_kmd_4node_2gpu_if.json \
  --driver-module tmp/unpatched-kmd-20261015/amdgpu.ko \
  --driver-package tmp/amdgpu-dkms_7.1.3.31500000-2390945.24.04_all.deb \
  --rank-binary tmp/p3v-alias-observation-release/tests/hip_rccl_gfx1250_test \
  --output /tmp/rj-review-qemu-retry-20261019 --timeout 600

python3 scripts/run-rccl-pod-emulated.py \
  --rocjitsu tmp/p3v-alias-observation-release/tools/rocjitsu/rocjitsu \
  --config configs/gfx1250_mi455x_kmd_4node_2gpu_if.json \
  --rank-binary tmp/p3v-alias-observation-release/tests/hip_rccl_gfx1250_test \
  --output /tmp/rj-review-emu-retry-20261019 --timeout 600
```

## Results and limits

Both modes ran AllReduce, Broadcast, AllGather, ReduceScatter, and SendRecv
at 64 and 262144 float elements, three iterations per case, with seed 79.
All eight ranks passed all five GoogleTest cases at both sizes. The cases
checked independent numerical references within their specified tolerances;
AllReduce logged zero result mismatches per iteration. The controller's
`numerical_mismatches: 0` field is inferred from those case successes and the
absence of nonzero mismatch diagnostics. The logs identify each rank's node
and selected local GPU index, but do not record a runtime GPU identity or
per-case, per-iteration mismatch totals for the other four cases. Each rank
and size selected both local `P2P/IPC` and cross-node
`NET/Socket` channels. In rank order, the P2P/IPC and NET channel counts per
size were `11/31`, `10/11`, `11/41`, `10/16`, `11/36`, `10/21`, `11/31`, and
`10/11` in both modes.

Each node's forward local route committed 34,350 writes and 17,309,208 bytes
in VFIO mode, or 29,676 writes and 14,949,384 bytes in emulated KFD mode.
The reverse route committed 549 writes and 5,136 bytes in VFIO mode, or
6,621 writes and 3,150,288 bytes in emulated KFD mode. The directed records
matched 34,899 VFIO or 36,297 emulated effects per node, with zero drops.
All nodes finished within the 600-second bound and left no VFIO or daemon
sockets. The cold lazy-cache fill regression also passed its isolated cache
tests; the broader serial cache, VM, VFIO, and transport selection passed
446/446 tests.
GuestKfd map and unmap regressions passed in Release and ASan/UBSan builds
for retries after partial progress and after full progress with a pending
final flush.

These observations support modeled local fabric behavior behind UC PCIe peer
mappings for the saved binary. A fresh two-mode run with recorded source state
and durable evidence is required to qualify the current tree. Native xGMI
discovery, cacheable peer coherence, default ROCr IPC mode, and the later
two-host gate remain unqualified.
