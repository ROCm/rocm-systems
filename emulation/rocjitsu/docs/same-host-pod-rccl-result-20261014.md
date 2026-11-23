# Same-host pod RCCL qualification, 2026-10-14

This historical run passed the original [same-host gate](same-host-pod-rccl-gate.md)
in both QEMU/VFIO and emulated KFD mode. The revised gate prohibits the nonstandard
QEMU/libvfio-user peer transport used by the VFIO run, so this record
no longer qualifies the VFIO mode under the current criteria. The emulated KFD
result remains valid for that mode. This result was produced from checkout
`07988db4b20d25ddf8cab50a008791255ec96532` with uncommitted working-tree
changes. No commit was created. The complete machine-readable records and logs
are preserved under `tmp/same-host-pod-20261014/evidence/`:

| Mode | Record | Record SHA-256 | Wall time | Node exits | Sockets left |
| --- | --- | --- | ---: | --- | ---: |
| QEMU/VFIO | `evidence/rj-pod-vfio-20261014-run5/result.json` | `8c7af4528674dbe0d342abee9ce38804204fc3977b2b8e875adb0a9f8723f3fe` | 115.81 s | `0,0,0,0` | 0 VFIO |
| Emulated KFD | `evidence/rj-pod-emu-20261014-run17/result.json` | `37d065eea8296ea42a0910b4fdb0c63bcd7c42e26be1b382697579dc05246253` | 58.29 s | `0,0,0,0` | 0 daemon |

Each record retains the exact generated node commands, runtime environment,
per-rank results, channel selections, directed route totals, and SHA-256 hashes
of every captured log. The original output directories are
`/tmp/rj-pod-vfio-20261014-run5` and `/tmp/rj-pod-emu-20261014-run17`.
The preserved copies have the same contents and hashes. VFIO guest and server
logs are under `nodeN/guest.log` and `nodeN/server.log`; emulated KFD rank logs
are under `nodeN/rankR-sizeS.log` with each node's `launcher.log`.

## Topology and driver path

`configs/gfx1250_mi455x_kmd_4node_2gpu_if.json` (SHA-256
`c14b21b6df774e9fddcba59c93f0c27f5063dcd086d39f38b360bdb0994fcd22`)
defines four nodes. Their GPU IDs are `(1250,1251)`, `(1252,1253)`,
`(1254,1255)`, and `(1256,1257)`. Each pair reaches a node-local Simdojo
Infinity Fabric switch through modeled GPU endpoints. Separate `net0` through
`net3` endpoints describe inter-node reachability; no GPU peer request uses a
NIC link. Four independent QEMU guests each expose two PCI GPUs and one NIC.
The guest NICs used `10.77.0.1` through `10.77.0.4` on QEMU socket multicast
segment `239.192.14.1:57550`. Four isolated emulated KFD daemons project the
same node pairs, use separate runtime directories and `NCCL_HOSTID` values,
and connect cross-node Socket traffic over host loopback.

The pinned gfx1250 A0 AMDGPU/AMDKFD driver uses its ordinary **PCIe peer BAR
mapping with a UC peer PTE**. RCCL selected `P2P/IPC` for local ranks and
`NET/Socket` for remote ranks. Local peer effects crossed the modeled
Infinity Fabric route behind the supported PCIe mapping. This is not evidence
of a native xGMI discovery or mapping path; the link class and `hive_id` alone
do not establish native xGMI. The emulated run set the public ROCr
`HSA_ENABLE_IPC_MODE_LEGACY=1` option. The default ROCr IPC mode is not
qualified by this record.

The guest `amdgpu.ko` was built from
`amdgpu-dkms_7.1.3.31500000-2390945.24.04_all.deb` with the driver bug-fix
patches `0001` through `0008` in numeric order. Optional diagnostic patch
`0011` was not applied. Omitted historical experimental guest patches were
also absent. These local patches are no longer included in this repository.
The module inside the final guest image is the staged override
described here; the older `base-image-lock.json` records the source image, not
the final module hash. The guest kernel was `6.17.0-23-generic`. QEMU came
from pinned revision `f8b2f64e2336a28bf0d50b6ef8a7d8c013e9bcf3` with
a nonstandard peer transport, PCIe AtomicOp capability, and XIO3130
LinkCap fixes. These host/QEMU changes added no guest-driver protocol. The
nonstandard transport is disallowed by the revised gate. The numerical and
route evidence below must not be presented as a stock-QEMU/libvfio-user
qualification.

| Input | SHA-256 |
| --- | --- |
| QEMU binary `tmp/p2v-qemu-v2-build/qemu-system-x86_64` | `826c251ea01336eeea622cd734e7881159ee7dab38c5a5092cd4e565ef2cfaf3` |
| Guest kernel `tmp/p3v-guest-coherent-trial/vmlinuz` | `e484d1fc35c787c76061191aecb146a851cf9cab37259665f61c9863f93a6194` |
| Final guest image `tmp/same-host-pod-20261014/initramfs-final.gz` | `a98cf55d04361f74f3e6f7d26e4e00f52c3ca6a4910cae8a5d7c44bd6e9f5052` |
| Staged guest `amdgpu.ko` | `58176b0957318fada1caefaa8e722aef2c171c50f9357728c442a3f578328d15` |
| DKMS source package | `edf6b629f2cb2d6091aac531c0787aee2387bdc0748a91f21ea38c6c10970ffb` |
| rocJITsu Release binary, both modes | `b58ceb968d7b41d120030f4ea9a2b3a8a8c7674b43afbc1a6056b2a46c7194f2` |
| Final `hip_rccl_gfx1250_test`, both modes and staged in guest | `e40759d0783758a77bdd96317a6825754c7656f4b35c199c04e18f66c2488c7a` |

The guest logged AMDGPU `7.1.3.31500000`, HIP `7.14.60850-0000000`, ROCr
`1.21`, and RCCL `2.30.4-HEAD:2b22ab0`. The staged RCCL library
`librccl.so.1.0` has SHA-256
`7cd64de6a62b52ed9a457b6d6b9158da49c35fcab44d036f028e84cfa56141a8`.
The ROCm runtime and RCCL packages are
`7.14.0~pre3-29052710811`; their complete package hashes are in
`root/etc/rocjitsu/workload-inventory.json` under
`tmp/same-host-pod-20261014/`. The guest staged header-only firmware
fixtures for this functional model. The emulator does not execute their
payloads. Their version fields identify these test inputs and do not describe
production microcode. The fields were parsed using the pinned driver's
`common_firmware_header`; `ip_discovery.bin` is a generated discovery table
without that header.

| Firmware fixture | Header / IP version | Fixture version field | SHA-256 |
| --- | --- | --- | --- |
| `gc_12_1_0_imu.bin` | 1.0 / 12.1 | `0x00000001` | `fcf3390da9dc58a9dca8842b8bc2ec48523c702fc5c1235c6143ded2b1221038` |
| `gc_12_1_0_mec.bin` | 2.0 / 12.1 | `0x00000001` | `db4006a1b46367027dbfcb674c2a3a2120e06314c08bad87a06aa754159b2142` |
| `gc_12_1_0_rlc_1.bin` | 2.0 / 12.1 | `0x00000001` | `e35cea989beb6601ff544b7c5c7101a1eb370aa19edd723eb1b94659616ac030` |
| `gc_12_1_0_uni_mes.bin` | 1.0 / 12.1 | `0x00000001` | `fa0abe49540f9be52a7f43998eee2c1a1db3f948468fef4a5509973d4b563aa1` |
| `sdma_7_1_0.bin` | 3.0 / 7.1 | `0x00000001` | `8258fbad36b70d7dea2100fa1f3dd90f996f2a83a5a8fe9f0bf401a895f8276d` |
| `ip_discovery.bin` | generated discovery table | n/a | `c871d51f36ee0b67f4784ba3c0196e6f4ed66ae36c9373b295d4af1a14c96358` |

## Exact controller commands

Both controllers used an 1800-second overall deadline. The guest used TCG,
four vCPUs, 12 GiB RAM, two VFIO-user sockets per node, the kernel arguments
and environment recorded in the VFIO result JSON, and `eth0` for NET. The
emulated controller used four daemons, separate runtime directories, the
standard ROCr legacy IPC option, and `lo` for NET.

```sh
python3 scripts/run-rccl-pod-vfio.py \
  --qemu tmp/p2v-qemu-v2-build/qemu-system-x86_64 \
  --kernel tmp/p3v-guest-coherent-trial/vmlinuz \
  --initramfs tmp/same-host-pod-20261014/initramfs-final.gz \
  --rocjitsu tmp/p3v-alias-observation-release/tools/rocjitsu/rocjitsu \
  --config configs/gfx1250_mi455x_kmd_4node_2gpu_if.json \
  --driver-module tmp/same-host-pod-20261014/root/modules/amdgpu.ko \
  --driver-package tmp/amdgpu-dkms_7.1.3.31500000-2390945.24.04_all.deb \
  --rank-binary tmp/p3v-alias-observation-release/tests/hip_rccl_gfx1250_test \
  --output /tmp/rj-pod-vfio-20261014-run5 --timeout 1800

python3 scripts/run-rccl-pod-emulated.py \
  --rocjitsu tmp/p3v-alias-observation-release/tools/rocjitsu/rocjitsu \
  --config configs/gfx1250_mi455x_kmd_4node_2gpu_if.json \
  --rank-binary tmp/p3v-alias-observation-release/tests/hip_rccl_gfx1250_test \
  --output /tmp/rj-pod-emu-20261014-run17 --timeout 1800
```

Each mode used one eight-rank communicator, two ranks per node, seed `79`,
`ROCR_VISIBLE_DEVICES=0` or `1` per rank, 64 and 262144 float elements, and
three iterations at each size for AllReduce, Broadcast, AllGather,
ReduceScatter, and SendRecv. The rank binary calculated host-side references
and compared every GPU result. All 16 rank/size executions in each mode passed
all five cases against tolerance-based numerical references. AllReduce
logged explicit zero mismatches. The verifier infers its zero-mismatch summary
for the other cases from passing checks and no nonzero diagnostics; their logs
do not provide per-case mismatch counts. The controller also required
`P2P/IPC` and `NET/Socket` channels in each rank/size log; a NET-only run
would fail.

| Rank (node) | Sizes and five cases, both modes | VFIO P2P/NET channel records per size | Emulated P2P/NET channel records per size |
| --- | --- | ---: | ---: |
| 0 (0) | 64, 262144: 3 iterations, all passed | 11 / 31 | 11 / 31 |
| 1 (0) | 64, 262144: 3 iterations, all passed | 10 / 11 | 10 / 11 |
| 2 (1) | 64, 262144: 3 iterations, all passed | 11 / 41 | 11 / 41 |
| 3 (1) | 64, 262144: 3 iterations, all passed | 10 / 16 | 10 / 16 |
| 4 (2) | 64, 262144: 3 iterations, all passed | 11 / 36 | 11 / 36 |
| 5 (2) | 64, 262144: 3 iterations, all passed | 10 / 21 | 10 / 21 |
| 6 (3) | 64, 262144: 3 iterations, all passed | 11 / 31 | 11 / 31 |
| 7 (3) | 64, 262144: 3 iterations, all passed | 10 / 11 | 10 / 11 |

## Directed fabric effects and teardown

Every expected GPU direction committed peer writes. Each result JSON checks
the individual effect trace against these route totals and requires zero
dropped records. All reported reads were zero; a committed write satisfies
the gate's peer read-or-write requirement.

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

Each VFIO node captured 34,899 effects, and each emulated KFD node captured
36,297; all eight node traces report zero drops. All ranks, guest launchers,
and emulated node helpers completed within their bounds. Every QEMU guest
reported its two AMDGPU-bound devices, numerical `result=PASS`, and orderly
poweroff. The four QEMU launchers and four emulated helpers exited zero. No
VFIO-user or emulated daemon sockets remained.

`FabricEndpoint*` passed 37 focused tests, including the prepared-maintenance
retry and one-commit regression. A serial CTest selection for fabric, VFIO,
PCI, VM, KFD, cache, MES, and queue components passed all 430 tests; the
output is `evidence/rj-pod-regressions-20261014.log`. The VFIO launcher Python
suite passed all 56 tests with Unix socket access; its output is
`evidence/rj-pod-launcher-unittest-20261014.log`. `gzip -t` on the final guest
image, `sh -n` on guest `init`, and `git diff --check` passed.

The two-host run, native xGMI driver path, default ROCr IPC mode, and
cacheable peer-coherence qualification remain separate follow-on work.
