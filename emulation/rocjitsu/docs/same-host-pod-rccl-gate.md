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

Run both controllers from `emulation/rocjitsu` in the `rocm-systems` checkout;
they resolve `scripts/` relative to the working directory. Use a fresh, short
`--output` path under `/tmp` because the daemon socket paths live below it.
Use `configs/gfx1250_mi455x_kmd_4node_2gpu_if.json` with
`scripts/run-rccl-pod-vfio.py` and `scripts/run-rccl-pod-emulated.py`. The
current controllers require exactly four nodes with two GPUs each, even though
the gate's topology contract allows at least two GPUs per node. They reject a
larger shape rather than selecting a subset. The [guest artifact recipe](../tests/guest/README.md#build-and-pack-the-guest-payloads)
identifies the external QEMU, driver, kernel, firmware, ROCm, and RCCL inputs,
packages the guest payload, and gives the current-build VFIO command. Supply
matching local artifacts before using that command. Run the equivalent
emulated KFD mode from the same directory with the current binaries:

```sh
python3 scripts/run-rccl-pod-emulated.py \
  --rocjitsu build-vfio/tools/rocjitsu/rocjitsu \
  --config configs/gfx1250_mi455x_kmd_4node_2gpu_if.json \
  --rank-binary build-vfio/tests/hip_rccl_gfx1250_test \
  --output /tmp/rocjitsu-pod-emulated --timeout 600
```

The [2026-10-19 result](same-host-pod-rccl-result-20261019.md) preserves
historical commands and input hashes, not a current-build recipe.

### Capture source and runtime provenance

Immediately before building the binaries for a fresh run, save the source
revision, tracked changes, and untracked rocJITsu source files. Run these
commands from `emulation/rocjitsu`, then repeat the capture after the build;
set `pod_capture_dir=/tmp/rocjitsu-pod-evidence/source-after-build` for the
second capture so it does not overwrite the first. If the source state changed,
rebuild from a stable source state and repeat both captures; the two snapshots
alone cannot show which contents the build consumed.

```sh
pod_capture_dir=/tmp/rocjitsu-pod-evidence/source-before-build
mkdir -p "$pod_capture_dir"
git -C ../.. rev-parse HEAD > "$pod_capture_dir/source-commit.txt"
git -C ../.. status --porcelain=v1 --untracked-files=all \
  > "$pod_capture_dir/source-status.txt"
git -C ../.. diff --binary HEAD > "$pod_capture_dir/source.patch"
git -C ../.. ls-files --others --exclude-standard -z -- emulation/rocjitsu \
  > "$pod_capture_dir/untracked.paths0"
tar -C ../.. --null \
  --files-from="$pod_capture_dir/untracked.paths0" \
  -cf "$pod_capture_dir/untracked.tar"
```

Archive any other untracked source named by `source-status.txt` if it affects
the build; the tar command above covers only rocJITsu's untracked files.

The controllers hash their executable, manifest, and logs in `result.json`;
the emulated controller also hashes itself and its node helper. Hash the VFIO
launcher and helpers, including the shared verification module, separately
because the VFIO `result.json` does not include them:

```sh
sha256sum scripts/run-rccl-pod-vfio.py scripts/rccl_pod_vfio.py \
  scripts/run-vfio-guest.py \
  > /tmp/rocjitsu-pod-evidence/vfio-scripts.sha256
```

Neither controller identifies the HIP, ROCr, and RCCL libraries actually
mapped by each rank. For the emulated run, take each rank PID from
`node*/launcher.log` while it is alive. After the libraries have loaded, save
its `/proc/<pid>/maps` and hash the mapped `libamdhip64`,
`libhsa-runtime64`, and `librccl` files:

```sh
rank_pid=12345  # Replace with a live PID from node*/launcher.log.
cp "/proc/$rank_pid/maps" \
  "/tmp/rocjitsu-pod-evidence/emulated-rank${rank_pid}.maps"
awk '$6 ~ /\/lib(rccl|amdhip64|hsa-runtime64)\.so/ { print $6 }' \
  "/tmp/rocjitsu-pod-evidence/emulated-rank${rank_pid}.maps" | \
  sort -u | xargs -r sha256sum \
  > "/tmp/rocjitsu-pod-evidence/emulated-rank${rank_pid}-libraries.sha256"
for library in libamdhip64 libhsa-runtime64 librccl; do
  grep -q "/${library}.so" \
    "/tmp/rocjitsu-pod-evidence/emulated-rank${rank_pid}-libraries.sha256" || \
    { echo "missing mapped $library for rank $rank_pid" >&2; exit 1; }
done
```

Repeat the capture while the rank is alive if any library is missing. Require
all three hashes for every rank in both modes.

For VFIO, capture the same map and hashes **inside each guest** after each
rank has loaded its libraries and before it exits. Add the probe to the staged
copy of `tests/guest/rccl_pod_init` after it starts both ranks. Its `$!` values
identify subshells, so find the two rank processes by matching `/proc/*/exe`
to `/opt/rocjitsu/bin/hip_rccl_gfx1250_test`; poll until each map contains all
three libraries. Use `/tmp` for temporary files inside the guest, then print
each rank's command line, map, and hash lines to the guest serial log before
poweroff. The host evidence directory in the example does not exist inside
the guest. Rebuild and hash the instrumented initramfs. Do not substitute
host library hashes for guest mappings. Preserve the host emulated-run evidence
and the separately staged VFIO guest inputs, `result.json`, rank logs, and
fabric traces together in a durable archive.

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

Broadcast, AllGather, and SendRecv currently compare their copy-only results
with `EXPECT_NEAR` at `1e-4f`. These checks can pass a bit-different value, so
this gate establishes tolerance-based numerical agreement, not bit-exact
delivery for those three cases. Use exact comparisons before claiming that
stronger property.

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
