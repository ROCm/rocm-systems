# Same-host pod RCCL exit gate

This is the active same-host pod exit gate. It requires two equivalent runs:
QEMU/VFIO guests using the pinned AMDGPU/AMDKFD driver, and emulated KFD
nodes. Earlier single-guest P2P, four-GPU, and two-guest NET controls remain
useful regressions but do not pass this gate.

## Topology and driver contract

Run four independent nodes on one physical host. In VFIO mode, each node is
its own QEMU guest and rocJITsu VMM. Each node has at least two modeled GPUs
and runs one RCCL process per GPU, for at least eight ranks in one
communicator. In emulated KFD mode, run the equivalent four isolated node
projections, at least two GPUs and one process per GPU on each node. Give each
node its own runtime state and a distinct network identity. This topology is a
simulator workload fixture, not a claim about production pod wiring.

Connect GPUs *within* a node through modeled, modular Simdojo Infinity Fabric
hardware endpoints and links. Connect nodes through guest NICs in VFIO mode
and node-local network endpoints in emulated KFD mode. The GPU fabric must not
route through, or be identified with, the NIC network. Require a real local
peer read or write on every expected directed GPU route, with committed count
and byte totals. A topology description, `hive_id`, or successful peer map
alone is insufficient.

The selected gfx1250 A0 guest uses the pinned driver's ordinary PCIe peer BAR
and UC PTE path. The simulated local fabric carries the admitted peer memory
effect behind that driver-visible path. Describe RCCL's observed local channel
as P2P/IPC over PCIe peer BAR mappings when that is what the driver and logs
show. Do not call it native xGMI merely because the model has Infinity Fabric
links or the device configuration has `hive_id`. Native xGMI may be claimed
only after its discovery, mapping, and traffic path are shown through the
stock driver's supported interfaces.

Use the pinned AMDGPU/AMDKFD driver without local patches for the current
gate. Do not introduce a simulator-specific guest-driver protocol. Any
future KMD change must fix a driver defect or
improve a hardware-independent driver contract that can be submitted upstream;
record its rationale and independent verification. A simulator-only
accommodation does not belong in AMDGPU/AMDKFD. Missing IOMMU,
memory-controller, MMHUB, HDP, UTC/TLB, or related behavior belongs in modular
hardware models reached through the stock register, packet, and discovery
interfaces.

The QEMU/VFIO run must also use stock QEMU and libvfio-user peer transport
interfaces. A rocJITsu-specific extension to `VFIO_USER_DMA_MAP` does not
satisfy this gate. Peer BAR access must retain its local fabric effects and
fail-closed lifetime checks through a standard transport path. A NET-only
fallback does not satisfy this requirement.

## Run this gate

Use `configs/gfx1250_mi455x_kmd_4node_2gpu_if.json` with
`scripts/run-rccl-pod-vfio.py` and `scripts/run-rccl-pod-emulated.py`. The
current controllers require exactly four nodes with two GPUs each, even though
the gate's topology contract allows at least two GPUs per node. They reject a
larger shape rather than selecting a subset. The [guest artifact recipe](../tests/guest/README.md#build-and-pack-the-guest-payloads)
identifies the external QEMU, driver, kernel, firmware, ROCm, and RCCL inputs,
packages the guest payload, and gives the current-build VFIO command. Supply
matching local artifacts before using that command. Run the equivalent
emulated KFD mode from the repository root with the current binaries:

```sh
python3 scripts/run-rccl-pod-emulated.py \
  --rocjitsu build-vfio/tools/rocjitsu/rocjitsu \
  --config configs/gfx1250_mi455x_kmd_4node_2gpu_if.json \
  --rank-binary build-vfio/tests/hip_rccl_gfx1250_test \
  --output /tmp/rocjitsu-pod-emulated --timeout 600
```

The [2026-10-19 result](same-host-pod-rccl-result-20261019.md) preserves
historical commands and input hashes, not a current-build recipe.

## Workload and passing evidence

For **both** modes, run AllReduce, Broadcast, AllGather, ReduceScatter, and
SendRecv at a recorded small and large message size. Repeat each size enough
times to expose intermittent failures. Every rank must check every result
against an independently calculated numerical reference. Record each rank's
node and selected local GPU index, case, size, and iteration. Require all five
GoogleTest cases to pass for each rank and size, with no nonzero mismatch
diagnostic; AllReduce also logs an explicit result mismatch count per
iteration. Require the same eight-rank communicator to show both intra-node
RCCL P2P/IPC channels and inter-node
`NET/Socket` channels. A NET-only pass fails the gate.

Save rank logs and channel selections, plus fabric-effect traces identifying
each expected directed local GPU route, committed read/write counts, and
bytes. Check that trace accounting matches the summary and that no effects
are dropped. Bound each rank, guest, VMM, and the complete run; require clean
rank and guest teardown, zero launcher status, and no remaining VFIO sockets.
Record the exact manifest, guest network layout, commands and environment,
driver source/package and module hashes, firmware and user-space versions,
QEMU and rocJITsu binary hashes, seed, message sizes, iteration count,
per-rank case results, channel evidence, route totals, and complete logs.

The two-host cluster run and cacheable peer-coherence qualification are later
gates. The selected A0 peer PTE is UC, so this gate does not require a
cacheable peer mapping. Do not mark it complete from topology enumeration,
peer-map success, a NET-only RCCL run, or an earlier single-guest result.

## Qualification status

The 2026-10-19 [local result](same-host-pod-rccl-result-20261019.md) records
passing per-rank checks, local fabric effects, NET channels, and clean teardown
in both modes for its saved binary. Its embedded revision identifies a reported
checkout, but no build-time working-tree status or patch was saved. The result
therefore does not qualify the current rebased source. Its
`numerical_mismatches: 0` field is inferred from case success and the absence
of nonzero mismatch diagnostics, not measured for every case and iteration.
The saved logs identify the selected local GPU index, but do not record a
runtime GPU identity or a mismatch count for every case and iteration.

The gate remains open for the current tree. Run both modes again, capture the
exact source revision and any working-tree patch, hash the executed binaries
and loaded runtime libraries, and archive the commands, rank logs, and traces.

The 2026-10-15 [qualification record](same-host-pod-rccl-result-20261015.md)
documents an earlier pass in both modes. Its QEMU/VFIO guests used a
zero-local-patch AMDGPU/AMDKFD module and standard VFIO-user peer DMA maps,
with only general QEMU PCIe and libvfio-user BAR bug fixes. All eight ranks
passed the numerical, local P2P/IPC, cross-node NET, directed fabric-effect,
bounded-completion, and teardown checks in both modes.

The earlier [2026-10-14 record](same-host-pod-rccl-result-20261014.md) and
[zero-patch driver ablation](unpatched-amdgpu-pod-ablation-20261015.md) are
historical controls. Their VFIO runs used a removed nonstandard peer
transport. This gate does not qualify native xGMI, default ROCr IPC mode,
cacheable peer coherence, or the later two-host run.
