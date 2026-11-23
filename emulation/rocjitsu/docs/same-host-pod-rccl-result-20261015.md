# Same-host pod RCCL qualification, 2026-10-15

The [same-host gate](same-host-pod-rccl-gate.md) passed in QEMU/VFIO and
emulated KFD modes. Four independent nodes ran on one physical host. Each
node contained two modeled gfx1250 A0 GPUs, and each GPU ran one RCCL rank in
the same eight-rank communicator. Local GPU traffic traversed modeled
node-local Infinity Fabric endpoints and links. Inter-node RCCL traffic used
`NET/Socket` over the guest NICs or, in emulated KFD mode, the node network
endpoints on host loopback. GPU VRAM was separate per GPU and per node; the
collectives did not synchronize through one shared HBM allocation.

The QEMU guest loaded an AMDGPU/AMDKFD module built from the pinned DKMS
package with **zero local driver patches**. The pinned driver's observed local
peer path was UC PCIe peer BAR mapping. RCCL used `P2P/IPC` locally and
`NET/Socket` across guests. The node-local fabric carried the local peer
memory effects behind that PCIe mapping. This does not establish native xGMI
discovery, mapping, or traffic. The emulated KFD run selected ROCr's public
`HSA_ENABLE_IPC_MODE_LEGACY=1` option; default IPC mode is not qualified.

The checkout was at `07988db4b20d25ddf8cab50a008791255ec96532` with
uncommitted changes. No commit was created. The complete records and logs
are preserved under `tmp/same-host-stock-transport-20261015/evidence/`.

| Run | Result JSON SHA-256 | Time | Node exits | Sockets left |
| --- | --- | ---: | --- | ---: |
| QEMU/VFIO `rj-u0-pod-stock-qemu-20261015` | `3fba6399791af79c8fc22627e084650f82fa0c749f9e17dfd2ba8bf369396eaa` | 94.31 s | `0,0,0,0` | 0 VFIO |
| Emulated KFD `rj-u0-pod-emu-stock-20261015` | `527c2d4fa6a0c6e249f27a5de4ae0beca407a943ff6ccb55104d34808c8c335f` | 59.28 s | `0,0,0,0` | 0 daemon |

Both controllers used an 1800-second overall bound. Their `result.json`
files contain the exact generated node commands, selected environment,
per-rank results, channel records, route totals, socket checks, and SHA-256
hashes of every captured log. The preserved copies have the same result JSON
hashes as the original runs in `/tmp/`.

## Transport and software inputs

The four QEMU guests each had four vCPUs, 12 GiB RAM, two VFIO-user PCI GPUs,
and one `virtio-net-pci` NIC. Their NICs used `10.77.0.1` through
`10.77.0.4` on multicast segment `239.192.14.1:46617`. QEMU was built from
10.1.0 commit `f8b2f64e2336a28bf0d50b6ef8a7d8c013e9bcf3` with only
three general PCIe fixes: preserve the VFIO-user PCI class code, give the
XIO3130 downstream port a valid LinkCap, and advertise opt-in root AtomicOp
completion and switch routing for implemented guest-RAM atomics. There is no
rocJITsu-specific VFIO-user command, trailer, capability, or QEMU device
property. Standard descriptorless `VFIO_USER_DMA_MAP` messages identify peer
BAR IOVAs; the server resolves them against its live PCI BAR decoder.

The server used libvfio-user source revision
`37491ed9af828fc161238dacd82e83ea35a09f87` with only the generic
sub-4-GiB 64-bit BAR upper-address mask fix. That fix is required for the
guest's peer BAR address to survive configuration writes; it is a library
bug-fix candidate, not a private transport. A pre-fix regression observed a
zeroed upper BAR register and a failed peer read; the fixed regression and
standard-transport two-GPU guest probe passed.

| Input | SHA-256 |
| --- | --- |
| System manifest `configs/gfx1250_mi455x_kmd_4node_2gpu_if.json` | `c14b21b6df774e9fddcba59c93f0c27f5063dcd086d39f38b360bdb0994fcd22` |
| QEMU `tmp/p2v-qemu-stock-transport-build/qemu-system-x86_64` | `3aab60dfb33fe2dac93e89b758f480f9758dfc633b8854363276854d9e132c23` |
| rocJITsu `tmp/p3v-alias-observation-release/tools/rocjitsu/rocjitsu` | `04b41b46cc1cec8d0e97afd54e35dd1f6af16342a97f44465efa62cab83ae8a8` |
| Guest kernel `tmp/p3v-guest-coherent-trial/vmlinuz` | `e484d1fc35c787c76061191aecb146a851cf9cab37259665f61c9863f93a6194` |
| Guest image `tmp/unpatched-kmd-20261015/pod-initramfs.gz` | `0056f623c9eadc42065549114a4b1f5c539f3124dac18449bd67fa15fc5003ce` |
| Zero-patch guest `amdgpu.ko` | `ffe029d3f994de1d8b5c0ce044854513dd6d31969f562b9fb4e2f374d41c7d13` |
| AMDGPU DKMS source package | `edf6b629f2cb2d6091aac531c0787aee2387bdc0748a91f21ea38c6c10970ffb` |
| RCCL rank binary `hip_rccl_gfx1250_test` | `e40759d0783758a77bdd96317a6825754c7656f4b35c199c04e18f66c2488c7a` |
| Staged `librccl.so.1.0` | `7cd64de6a62b52ed9a457b6d6b9158da49c35fcab44d036f028e84cfa56141a8` |
| libvfio-user 64-bit BAR fix | `2c537fc975c95e3b29b10bdd87a25626de6143ef8fe11b6b7b2736850e386c64` |

The three QEMU patch hashes, in class-code, LinkCap, and AtomicOp order, are
`31f1f20a4de3534dc4dc84712ed9d76c8558d2d370ea094303864bd3dc7781df`,
`f84ad25980db13ab2df701cf9df51bbf133e01224f317a37efce801a547bf87f`,
and `eda4b47b2af4a5644c9f467583cfe6988f16bcdd274883613a880b5cab3bc6fc`.
The guest logged AMDGPU `7.1.3.31500000`, HIP
`7.14.60850-0000000`, ROCr `1.21`, and RCCL
`2.30.4-HEAD:2b22ab0`. The guest's
`etc/rocjitsu/workload-inventory.json` has SHA-256
`4096e9583aa58c6b0b6f5f7f60bd81d2e477cadb5be587e4fc4fe7354fcdc119`.
The firmware hashes match the staged inputs in the
[earlier result](same-host-pod-rccl-result-20261014.md); the generated
`ip_discovery.bin` is
`c871d51f36ee0b67f4784ba3c0196e6f4ed66ae36c9373b295d4af1a14c96358`.
The old image lock describes the source image; the guest's printed
`driver_sha256` attests to the replacement zero-patch module in every node.

## Exact controller commands

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
  --output /tmp/rj-u0-pod-stock-qemu-20261015 --timeout 1800

python3 scripts/run-rccl-pod-emulated.py \
  --rocjitsu tmp/p3v-alias-observation-release/tools/rocjitsu/rocjitsu \
  --config configs/gfx1250_mi455x_kmd_4node_2gpu_if.json \
  --rank-binary tmp/p3v-alias-observation-release/tests/hip_rccl_gfx1250_test \
  --output /tmp/rj-u0-pod-emu-stock-20261015 --timeout 1800
```

The controllers used seed `79`, one rank per GPU, `ROCR_VISIBLE_DEVICES=0`
or `1`, 64 and 262144 float elements, and three iterations per case at each
size. Every rank calculated an independent host-side numerical reference for
AllReduce, Broadcast, AllGather, ReduceScatter, and SendRecv. The emulated
controller ran four isolated daemon processes with distinct runtime
directories and `NCCL_HOSTID` values, used `lo` for cross-node Socket
traffic, and enabled the public legacy ROCr IPC option. Every rank/size log
contained both local `P2P/IPC` and remote `NET/Socket` channels. All five
GoogleTest cases passed at both sizes on every rank with tolerance-based
reference checks. AllReduce explicitly logged zero result mismatches per
iteration. The controller's `numerical_mismatches: 0` field is inferred from
passing cases and no nonzero mismatch diagnostics; the logs do not give a
runtime GPU identity or mismatch totals for every case and iteration. They
record each rank's node and selected local GPU index.

| Rank (node) | Sizes and case results in both modes | P2P/IPC and NET channel records per size |
| --- | --- | ---: |
| 0 (0) | 64 and 262144; five cases × three iterations passed | 11 / 31 |
| 1 (0) | same | 10 / 11 |
| 2 (1) | same | 11 / 41 |
| 3 (1) | same | 10 / 16 |
| 4 (2) | same | 11 / 36 |
| 5 (2) | same | 10 / 21 |
| 6 (3) | same | 11 / 31 |
| 7 (3) | same | 10 / 11 |

## Directed effects, teardown, and regressions

Each node's forward and reverse local GPU route committed peer writes.
Individual effect records matched the route totals and reported zero drops.
There were no peer reads in this RCCL run; the separate two-GPU guest probe
read and wrote GPU0 from GPU1, checked the numerical result, and recorded
one four-byte read and one four-byte write on route `1251 -> 1250`.

| GPU route | VFIO writes / bytes | Emulated KFD writes / bytes |
| --- | ---: | ---: |
| 1250 → 1251 | 34,350 / 17,309,208 | 29,676 / 14,949,384 |
| 1251 → 1250 | 549 / 5,136 | 6,621 / 3,150,288 |
| 1252 → 1253 | 34,350 / 17,309,208 | 29,676 / 14,949,384 |
| 1253 → 1252 | 549 / 5,136 | 6,621 / 3,150,288 |
| 1254 → 1255 | 34,350 / 17,309,208 | 29,676 / 14,949,384 |
| 1255 → 1254 | 549 / 5,136 | 6,621 / 3,150,288 |
| 1256 → 1257 | 34,350 / 17,309,208 | 29,676 / 14,949,384 |
| 1257 → 1256 | 549 / 5,136 | 6,621 / 3,150,288 |

The VFIO nodes captured 34,899 effects each; emulated nodes captured 36,297
each. All eight node processes exited zero within their bound. Guest logs
reported both AMDGPU-bound devices and orderly completion, with no `BUG:`,
`WARNING:`, `Call Trace:`, `FATAL:`, `UBSAN:`, or numerical failure. RCCL did
print a nonfatal warning that AMD SMI fabric support was disabled; no native
xGMI path is claimed. No VFIO-user or daemon socket remained. The focused
VFIO host and grant regression selection passed 42/42 tests. The revised
VFIO launcher suite passed 54/54 tests. `git diff --check`, `bash -n` on the
QEMU fix applicator, and Python compilation of both controllers passed.

This qualifies the smaller same-host pod-style configuration with UC PCIe
peer mappings over the modeled local fabric. Native xGMI driver discovery,
cacheable peer coherence, default ROCr IPC mode, and the two-host run are
separate gates.
