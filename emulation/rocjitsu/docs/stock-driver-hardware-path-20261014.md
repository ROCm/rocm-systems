# Stock-driver hardware-path same-host record

Date: 2026-10-14. Checkout: `users/agutierr/p4v-kfd-map-trace` at
`07988db4b20d25ddf8cab50a008791255ec96532`, with the working-tree
changes described in `docs/qemu-vfio.md`. No commit was created.

## Guest driver and image

The source package was
`tmp/amdgpu-dkms_7.1.3.31500000-2390945.24.04_all.deb` (SHA-256
`edf6b629f2cb2d6091aac531c0787aee2387bdc0748a91f21ea38c6c10970ffb`).
The freshly extracted DKMS source received local driver bug-fix patches
`0001` through `0008` in numeric order. Those patches are no longer included
in this repository. `patch-application.log` records their checks and
application; `build.log` records the module build against
`/usr/src/linux-headers-6.17.0-23-generic`. Patch `0011` is an optional
diagnostic and was not applied to this production module. Other historical
experimental patches were also absent. The scratch DKMS source acquired
temporary printk diagnostics after the production
module was staged; it is not the source of the module inside this image.

The removed fixes addressed independent VM-update and KFD retry error paths
in this pinned driver: link a page table only after its clear succeeds,
correct SDMA dependency fence ownership, abort unsubmitted failed PT/PDE
jobs, publish CPU TLB sequence after HDP flush, retry a failed compute TLB
flush, and retain KFD unmap and TLB-flush work for retry. The later zero-patch
UC workload did not exercise those failures. For this tested driver version,
these controls also did not establish SVM invalidation acknowledgement or
error ownership across failures for cacheable peer mappings. That contract
remained open for the later cacheable-coherence work.

| Input or output | SHA-256 |
| --- | --- |
| `tmp/stock-driver-eightpatch-20261014-serial/root/modules/amdgpu.ko` | `58176b0957318fada1caefaa8e722aef2c171c50f9357728c442a3f578328d15` |
| `tmp/stock-driver-eightpatch-20261014-serial/initramfs.gz` | `3bff122457a01f511fc35d4ec02972431141896cb0dc64ae10d59b26f6ca4cc2` |
| `tmp/stock-driver-eightpatch-20261014-serial/teardown-initramfs.gz` | `01d22291bf307509014d91bdf500f98133120c694a1577714fa58a9ec6faf7aa` |
| `tmp/p3v-guest-coherent-trial/vmlinuz` | `e484d1fc35c787c76061191aecb146a851cf9cab37259665f61c9863f93a6194` |
| Historical two-GPU fixture input | `05549fa4cf3397f2cc2923aaa46b273c5f131efa53e57531a5ef9b3cb367baf1` |

Extraction of `modules/amdgpu.ko` directly from `teardown-initramfs.gz`
gave the same SHA-256 as the staged production module. Extraction of `init`
showed `guest: done`, then `sleep 3`, then `poweroff -f`. The delay permits
asynchronous KFD process release to finish before forced test shutdown; it
does not alter the driver. `gzip -t` and `bash -n` passed.

## QEMU and server

The QEMU binary was built from the pinned QEMU 10.1 source revision
`f8b2f64e2336a28bf0d50b6ef8a7d8c013e9bcf3`. This historical build
used a nonstandard peer transport, opt-in PCIe AtomicOp root and switch
properties, and the XIO3130 downstream LinkCap fix. Application was
checked on a clean pinned source worktree at `tmp/qemu-atomic-clean-source/`;
its tracked diff passed `git diff --check`. The binary used in this run came
from `tmp/p2v-qemu-zero-validate/` with those same property and link changes.
Its root AtomicOp and switch routing properties report `default: off`.

| Binary | SHA-256 |
| --- | --- |
| `tmp/p2v-qemu-v2-build/qemu-system-x86_64` | `826c251ea01336eeea622cd734e7881159ee7dab38c5a5092cd4e565ef2cfaf3` |
| `tmp/p3v-alias-observation-release/tools/rocjitsu/rocjitsu` | `9b0d3470ab8bc6ded71d36e6d4ba8fcc88946834889aefa9ac421e513f58b4b3` |

The server is a Release VFIO-enabled build. The final build command was:

```sh
cmake --build tmp/p3v-alias-observation-release \
  --target rocjitsu_bin rocjitsu_model_only_test -j4
```

## Same-host guest result

The exact launch command, including the TCG, two VFIO-user sockets, a schematic
topology, four guest CPUs, expected numerical values, and opt-in root
AtomicOp capabilities, is in `run-configured-peer.sh`. It was run as:

```sh
bash tmp/stock-driver-eightpatch-20261014-serial/run-configured-peer.sh \
  /tmp/rj-stock-peer-final-20261014
```

The launcher exited zero. Both PCI functions bound to `amdgpu`; HIP reported
two GPUs and admitted GPU1 access to GPU0. The requester kernel read
`0x13579bdf` and wrote `0x1357c047` to the owner. The guest reported
`result=PASS`, freed both allocations, disabled peer access, and printed
`guest: done`. The server reported one committed GPU1-to-GPU0 read and one
write, four bytes each. Neither copied log contains `UBSAN:`, `WARNING:`,
`BUG:`, `Call Trace:`, `guest: FATAL:`, or `result=FAIL`.

| Log | SHA-256 |
| --- | --- |
| `tmp/stock-driver-eightpatch-20261014-serial/final-guest.log` | `8bd44f9c591e977743b59aba6f3e9235f41a6765ad57ab09d3828f301d538123` |
| `tmp/stock-driver-eightpatch-20261014-serial/final-server.log` | `c5efd41d47c18f620c90796bad4f46608c369e7776da75169481757c3257e51a` |

A control launch without root and switch AtomicOp properties bound both
PCI functions but KFD skipped them with `PCI rejects atomics 1<2090` and
HIP reported no ROCm-capable device. Its copied logs are
`tmp/stock-driver-eightpatch-20261014-serial/noatom-guest.log` (SHA-256
`3f6b4cf85c7acc4d62bcb32f24321414d18608f2d8968d12c1ed797505118b9c`)
and `tmp/stock-driver-eightpatch-20261014-serial/noatom-server.log` (SHA-256
`c6f2017b2cbb2badb9ba4534b968417a4a4dfc0b9ea133d38448f3b1ff15c6b5`).

## Focused verification

The launcher suite passed all 56 tests:

```sh
python3 -m unittest tests.scripts.test_run_vfio_guest -v
```

The serial CTest selections passed 70 VFIO/VM tests and 201 PCI hardware,
GART, VM fault, MES, and queue tests, respectively. Full output is in
`tmp/stock-driver-eightpatch-20261014-serial/focused-ctest.log` and
`tmp/stock-driver-eightpatch-20261014-serial/hardware-ctest.log`.

```sh
ctest --test-dir tmp/p3v-alias-observation-release --output-on-failure -j1 \
  -R '^(VfioTargetGrants\.|VfioDeviceHost\.|GpuPciDevice\.|GpuVmService\.|MesEngine\.|KfdIoctl\.|GpuQueueRegistry\.)'
ctest --test-dir tmp/p3v-alias-observation-release --output-on-failure -j1 \
  -R '^(GpuDevice\.|GpuDeviceHdpFlush\.|GpuDeviceDelivery\.|GpuDeviceMes\.|PciPhysicalMemoryAccess\.|GpuVmTranslation\.|GpuVmPipeline\.|GpuQueueRegistryTest\.|MesEngineStateTest\.|MesEngineTest\.|InterruptSinkTest\.)'
```

This checks ordinary peer reads and writes on one host using the driver's
normal interfaces. Peer BAR atomics are rejected because the peer endpoint
does not advertise completion. The observed stock-driver peer PTE is UC;
cacheable peer coherence and the two-host exit gate remain unqualified.
The nonstandard peer transport was confined to the host. It does not
qualify the current standard-transport gate.

## Cacheability boundary checked after this control

The selected guest configuration has `revision_id=1`. In the pinned DKMS
source, `amd/amdgpu/gmc_v12_1.c` selects `MTYPE_UC` for remote VRAM on GC
12.1.0 A0 (`gmc_v12_1_get_coherence_flags`, around lines 545-580). The KFD
mapping passes through `gmc_v12_1_get_vm_pte`; no simulator-specific driver
change is needed to obtain this policy. The GFX12 PTE decoder reads bits 54-55
and maps the UC encoding to `Mtype::UC`.

Translated vector reads whose full cache line maps to local VRAM now use the
existing Simdojo V$ and L2 components. Each clean tag keeps the exact
address-space handle, physical line, raw PTE memory type, permissions, and
local backing epoch. A hit checks its live transport session and VM namespace;
local VRAM writes, HDP flushes, and fabric maintenance retire stale clean
tags. Translated scalar reads and vector stores still use direct VM access.
The existing same-host peer guest checks ordinary transport and backing
effects; it does not demonstrate an owner cache hit or a cacheable peer alias.
The VFIO peer adapter rejects a
nonzero-VMID PTE with a memory type other than UC before submitting its
fabric effect. The VM translation also retains the raw GFX12 PTE memory-type
encoding so an NC-to-RW change cannot pass a deferred rewalk merely because
both encodings currently share the broad `Mtype::RW` cache class. VMID-0 GART
requests retain their uncached MES/SDMA route.

The focused `VfioTargetGrants.Gfx12GartPeerReadWriteUsesFabricWithStockPageTableWalks`
test used the real GFX12 PTE decoder and a live two-function VFIO grant. It
passed after checking that process-VMID RW peer reads and writes produce no
effect, a UC peer read produces one effect and the expected bytes, and exact
UNMAP denies a later read. The test was run from
`tmp/p3v-alias-observation-release/tests/rocjitsu_tests` after rebuilding
`rocjitsu_tests` and `rocjitsu_bin`; the local socket test required the
unsandboxed test runner. All 33 `VfioTargetGrants` and `VfioDeviceHost`
CTest cases passed in 2.16 seconds. The
`GpuVmTranslation.Gfx12PeerRewalkDistinguishesNcFromRwPteEncoding` test also
passed, proving that a PTE type change with the same broad cache class and
physical address is rejected on rewalk.

The unchanged stock-driver guest image then ran the same-host two-GPU script
against the rebuilt server:

```sh
bash tmp/stock-driver-eightpatch-20261014-serial/run-configured-peer.sh \
  /tmp/rj-stock-peer-raw-pte-20261014
```

The launcher exited zero. The guest reported the same numerical PASS and
`guest: done`, with no `guest: FATAL:`, `result=FAIL`, `UBSAN:`, `WARNING:`,
`BUG:`, or `Call Trace:`. The server recorded one four-byte GPU1-to-GPU0
read and one four-byte write. Artifact hashes are:

| Input or output | SHA-256 |
| --- | --- |
| `tmp/p3v-alias-observation-release/tools/rocjitsu/rocjitsu` | `04c60a924c6378c32a18c147ffb852d9032f87676e83d89acfbc9174fcaeca3c` |
| `/tmp/rj-stock-peer-raw-pte-20261014/guest.log` | `1afd5490245759e01d7417a09c82aa7b3d7b6adae33fc8dc64dbeaa56ad20785` |
| `/tmp/rj-stock-peer-raw-pte-20261014/server.log` | `5ababb12d18f8204bea4fda5c869cf0de1be67d006b4f0b42813ac172af76bd1` |

Cacheable coherence remains unqualified. A truthful follow-on experiment
needs a real hardware revision and mapping mode whose stock driver emits a
peer PTE with demonstrated cacheable semantics, then a peer translated cache
path tied to the process binding and the hub/TLB invalidation acknowledgment.
The VFIO grant and canonical backing must remain live through cache fill,
fabric maintenance, effect, completion, and retirement. The current A0
configuration has no such peer PTE. The available same-host topology cannot
qualify the separate-host exit gate.

## Translated local cache and same-host rerun

The local vector-read path uses the existing V$ and L2 cache components with
an exact address-space namespace. It remains restricted to complete local
cache-line translations and a PCI backing that advertises local write and HDP
publication. The functional visibility epoch retires clean local tags after
trapped VRAM writes, SDMA local writes, HDP flushes, and fabric maintenance.
An SDMA write can publish while it holds cache maintenance; publication does
not recursively acquire the domain lock. Scalar reads, vector stores, and
cacheable peer PTEs remain outside this path.

The rebuilt `rocjitsu_tests` and `rocjitsu_bin` targets passed the focused
`TranslatedLocalCache.ExactHandlePtePolicyAndFabricBoundaryRefetchOwnerData`,
`GpuDevice.AcceptsTheHdpFlushTheDriverIssues`,
`GpuDeviceSdma.RegisterBackedQueueRoutesAddressesOutsideTheGartApertureToVram`,
and both `GpuDeviceHdpFlush` tests. The selected PCI, VM, MES, KFD, SDMA, and
cache regression run passed 206 of 237 cases inside the sandbox; the other 31
were VFIO cases whose Unix socket creation was denied with `Operation not
permitted`. All 33 `VfioDeviceHost` and `VfioTargetGrants` cases passed when
rerun with local Unix socket access. `clang-format --dry-run --Werror` on the
changed cache implementation and PCI test, and `git diff --check`, passed.

The unchanged eight-patch stock-driver guest image was launched on this host
with the rebuilt server:

```sh
bash tmp/stock-driver-eightpatch-20261014-serial/run-configured-peer.sh \
  /tmp/rj-stock-peer-translated-local-20261014
```

The launcher exited zero. The guest reported two GPUs, numerical
`observed=0x13579bdf final_owner=0x1357c047 result=PASS`, and `guest: done`.
The server reported one committed GPU1-to-GPU0 read and one write of four
bytes each. Neither log contains `UBSAN:`, `WARNING:`, `BUG:`, `Call Trace:`,
`guest: FATAL:`, or `result=FAIL`.

| Input or output | SHA-256 |
| --- | --- |
| `tmp/p3v-alias-observation-release/tools/rocjitsu/rocjitsu` | `f145de91bec1d7ecd8b1220d555df998e7ac21e0611b25b6ef8fde721d6cc49e` |
| `/tmp/rj-stock-peer-translated-local-20261014/guest.log` | `83fce15e2a57df671400498237956b2b9fba63730bb4135149b1c71c7083c270` |
| `/tmp/rj-stock-peer-translated-local-20261014/server.log` | `ed1fc36d8acbca56a360006728a4ff129c5ac341cbdd2b6c031dea4ac41dbc37` |

This guest control exercises a UC peer mapping and does not prove an owner
cache hit followed by a peer write and owner refetch in the live guest. The
focused cache test covers that modeled boundary. Cacheable peer coherence,
grant-retirement denial after a cached fill, and separate-host behavior still
need their own qualification.
