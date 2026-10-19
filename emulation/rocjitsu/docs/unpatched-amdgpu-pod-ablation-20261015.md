# Zero-patch AMDGPU pod ablation, 2026-10-15

This is a driver-patch ablation control, **not** a pass of the current
[same-host pod gate](same-host-pod-rccl-gate.md). The guest module was built
from the pinned AMDGPU DKMS package with **zero local AMDGPU/AMDKFD patches**.
Both a two-GPU peer probe and a four-guest, eight-rank RCCL workload passed.
Both runs still used a nonstandard QEMU/libvfio-user peer transport,
which the current gate prohibits. The result shows that patches `0001` through
`0008` are unnecessary for these normal-path workloads under that transport;
it does not settle their independent driver error-path bugs. The subsequent
[same-host qualification](same-host-pod-rccl-result-20261015.md) passed with
the same zero-patch module and standard VFIO-user peer transport.

The checkout was at `07988db4b20d25ddf8cab50a008791255ec96532` with
uncommitted changes. No commit was created. Complete logs and result JSON are
preserved in `tmp/unpatched-kmd-20261015/evidence/`; the original runs were
`/tmp/rj-u0-peer-20261015-b` and `/tmp/rj-u0-pod-20261015`. The pod result
JSON has SHA-256
`77924d120487353e8f943b962b5559f168157392736eb7185d5a24c111385f46`
and retains every generated node command, selected environment value, rank
result, channel record, fabric total, socket check, and input/log hash.

## Inputs and commands

| Input | SHA-256 |
| --- | --- |
| `tmp/amdgpu-dkms_7.1.3.31500000-2390945.24.04_all.deb` | `edf6b629f2cb2d6091aac531c0787aee2387bdc0748a91f21ea38c6c10970ffb` |
| Zero-patch `tmp/unpatched-kmd-20261015/amdgpu.ko` | `ffe029d3f994de1d8b5c0ce044854513dd6d31969f562b9fb4e2f374d41c7d13` |
| Pod `tmp/unpatched-kmd-20261015/pod-initramfs.gz` | `0056f623c9eadc42065549114a4b1f5c539f3124dac18449bd67fa15fc5003ce` |
| QEMU `tmp/p2v-qemu-v2-build/qemu-system-x86_64` | `826c251ea01336eeea622cd734e7881159ee7dab38c5a5092cd4e565ef2cfaf3` |
| Kernel `tmp/p3v-guest-coherent-trial/vmlinuz` | `e484d1fc35c787c76061191aecb146a851cf9cab37259665f61c9863f93a6194` |
| rocJITsu `tmp/p3v-alias-observation-release/tools/rocjitsu/rocjitsu` | `b58ceb968d7b41d120030f4ea9a2b3a8a8c7674b43afbc1a6056b2a46c7194f2` |
| RCCL rank test `tmp/p3v-alias-observation-release/tests/hip_rccl_gfx1250_test` | `e40759d0783758a77bdd96317a6825754c7656f4b35c199c04e18f66c2488c7a` |
| `configs/gfx1250_mi455x_kmd_4node_2gpu_if.json` | `c14b21b6df774e9fddcba59c93f0c27f5063dcd086d39f38b360bdb0994fcd22` |

The zero-patch build used guest kernel headers `6.17.0-23-generic`; its full
output is `tmp/unpatched-kmd-20261015/build.log`. The pod image copied the
2026-10-14 workload root and replaced only `/modules/amdgpu.ko`. The module
inside that image has the same zero-patch hash. The source image includes the
same ROCm user space, RCCL, and model-compatible firmware recorded in the
[2026-10-14 result](same-host-pod-rccl-result-20261014.md). The old source
image's lock file does not attest to the replacement module; use the hash above
and the guest's printed `driver_sha256` for this run.

The two-GPU control was launched with:

```sh
bash tmp/unpatched-kmd-20261015/run-peer.sh /tmp/rj-u0-peer-20261015-b
```

Its guest reported the expected GPU1 read and owner write, `result=PASS`, and
`guest: done`. The server counted one committed read and one write on
`1251 -> 1250`, four bytes each. Neither log contains `WARNING:`, `BUG:`,
`FATAL:`, `Call Trace:`, or `result=FAIL`; no VFIO socket remained. The guest
and server log hashes are respectively
`0e5257821b220d4911352c3e0fd21e8b81fd2e12d51141d25b408ccf345b680c`
and
`9478468d94d238d12b19504ec6733491e8e3a71164946ad38412e43776f38be1`.

The pod control was launched with:

```sh
python3 scripts/run-rccl-pod-vfio.py \
  --qemu tmp/p2v-qemu-v2-build/qemu-system-x86_64 \
  --kernel tmp/p3v-guest-coherent-trial/vmlinuz \
  --initramfs tmp/unpatched-kmd-20261015/pod-initramfs.gz \
  --rocjitsu tmp/p3v-alias-observation-release/tools/rocjitsu/rocjitsu \
  --config configs/gfx1250_mi455x_kmd_4node_2gpu_if.json \
  --driver-module tmp/unpatched-kmd-20261015/amdgpu.ko \
  --driver-package tmp/amdgpu-dkms_7.1.3.31500000-2390945.24.04_all.deb \
  --rank-binary tmp/p3v-alias-observation-release/tests/hip_rccl_gfx1250_test \
  --output /tmp/rj-u0-pod-20261015 --timeout 1800
```

The controller generated four independent TCG guests, each with two VFIO-user
GPUs, four vCPUs, 12 GiB RAM, and an `eth0` NIC on multicast segment
`239.192.14.1:57734`. It used a nonstandard peer transport plus the standard
PCIe switch and AtomicOp capability model options. The generated command
lines and kernel arguments are recorded in `result.json`.
The guest driver had no simulator-specific protocol changes.

## Result and limits

All four guests exited zero in 94.3 seconds. Every rank passed AllReduce,
Broadcast, AllGather, ReduceScatter, and SendRecv against its numerical
reference at 64 and 262144 float elements, with three iterations per case and
zero mismatches. Every rank/size log included both local `P2P/IPC` and remote
`NET/Socket` channels.

| Rank | Node | P2P/IPC records per size | NET/Socket records per size |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 11 | 31 |
| 1 | 0 | 10 | 11 |
| 2 | 1 | 11 | 41 |
| 3 | 1 | 10 | 16 |
| 4 | 2 | 11 | 36 |
| 5 | 2 | 10 | 21 |
| 6 | 3 | 11 | 31 |
| 7 | 3 | 10 | 11 |

Each node's forward local GPU route committed 34,350 writes and 17,309,208
bytes. Each reverse route committed 549 writes and 5,136 bytes. There were no
peer reads in this RCCL run. The 34,899 captured effects per node matched the
route totals, with zero dropped. No guest or server log contains `WARNING:`,
`BUG:`, `FATAL:`, `Call Trace:`, or `result=FAIL`; no VFIO sockets remained.

The selected gfx1250 A0 driver uses UC PCIe peer BAR mappings for local
P2P/IPC. The modeled Infinity Fabric links carried the local effects behind
that driver-visible path; cross-guest traffic used the NIC and NET/Socket.
This does not demonstrate native xGMI discovery or mappings. The later
[standard-transport run](same-host-pod-rccl-result-20261015.md) qualified
both modes after the shared model changes.
