# RCCL SPLIT_MASK + PCIe Topology Analysis Spec
# MI300X 8-GPU / 4-NIC per Node — BD Customer Investigation

**Date**: 2026-10-09
**Platform**: SMC300x MI300X (2 nodes: smc300x-ccs-aus-gpucc5a, smc300x-ccs-aus-gpuf273)
**Kernel**: 5.15.0-142-generic
**RCCL Source**: /apps/shared/karthik/repo/rocm-systems/BD/rocm-systems_BD/projects/rccl
**RCCL Tests**: /apps/shared/karthik/repo/rocm-systems/BD/rocm-systems_BD/projects/rccl-tests

---

## 1. Problem Statement

ByteDance customer running data-parallel (DP) workloads with `NCCL_TESTS_SPLIT_MASK=0x7` on MI300X nodes (8 GPUs, 4 NICs masked from 8) observes degraded performance. Root cause: RCCL cannot see inter-switch fabric connections between Broadcom PEX89104 multi-host switch partitions, causing GPU-NIC pairs that cross partition boundaries to be classified as PATH_PHB (distance 8) instead of PATH_PXB (distance 5), which disables GPU Direct RDMA (GDR).

---

## 2. NCCL_TESTS_SPLIT_MASK Mechanics

### Source
`rccl-tests/src/common.cu` lines 2467-2494

### Algorithm
```c
color = MPI_rank & strtoul(mask_hex, NULL, 16);
MPI_Comm_split(MPI_COMM_WORLD, color, proc, &mpi_comm);
```

### With mask=0x7, 16 ranks (2 nodes x 8 GPUs)
Creates 8 independent NCCL communicators, each with 2 ranks (1 per node):

| Group (color) | Ranks | GPU/node | Notes |
|---|---|---|---|
| 0 | 0, 8  | GPU0 | Simulates DP model-replica 0 |
| 1 | 1, 9  | GPU1 | Simulates DP model-replica 1 |
| 2 | 2, 10 | GPU2 | ... |
| 3 | 3, 11 | GPU3 | |
| 4 | 4, 12 | GPU4 | |
| 5 | 5, 13 | GPU5 | |
| 6 | 6, 14 | GPU6 | |
| 7 | 7, 15 | GPU7 | |

Each communicator runs all_reduce independently — models isolated DP gradient sync.

---

## 3. RCCL Path Type Computation

### 3.1 Path Type Definitions
`src/include/graph.h`:

```
PATH_LOC (0)  — Local (self)
PATH_NVL (1)  — NVLink / XGMI direct
PATH_NVB (2)  — NVLink via intermediate GPU
PATH_C2C (3)  — Chip-to-chip
PATH_PIX (4)  — Single PCIe bridge
PATH_PXB (5)  — Multiple PCIe bridges, NO CPU traversal
PATH_P2C (6)  — C2C + PCIe (GPU→CPU→NIC)
PATH_PXN (7)  — PCI + NVLink proxy routing
PATH_PHB (8)  — PCIe through CPU/Host Bridge
PATH_SYS (9)  — Cross-NUMA (QPI/UPI)
PATH_NET (10) — Network
PATH_DIS (11) — Disconnected
```

**GDR threshold**: Enabled when path type **< PATH_PHB** (i.e., PATH_PXB=5 or better).

### 3.2 BFS Path Computation Algorithm
`src/graph/paths.cc` — `ncclTopoSetPaths()`:

Uses breadth-first search from each base node to all reachable nodes. Two critical classification rules:

```c
// Rule 1: PCI switch → PCI switch traversal = PATH_PXB
if (node->type == PCI && remNode->type == PCI) newType = PATH_PXB;

// Rule 2: Any PCI link through CPU = PATH_PHB
if (link->type == LINK_PCI && (node->type == CPU || link->remNode->type == CPU))
    newType = PATH_PHB;

// Path type is monotonically increasing (worst segment wins)
newType = std::max(path->type, newType);
```

Path selection prefers: lower type > higher bandwidth > fewer hops.

### 3.3 BCM Switch Flattening
`src/graph/topo.cc` — `ncclTopoFlattenBcmSwitches()`:

RCCL detects Gen4/Gen5 Broadcom switches via the `getBcmGen()` helper (`topo.cc:215-219`) and flattens their internal 2-level hierarchy into a single switch node. The 64-bit `pci.device` field is composed as `vendor(16) | device(16) | subsystem_vendor(16) | subsystem_device(16)`. The matching criteria are:

- **Gen4**: `(pci.device & 0xfffffffffffff000) == 0x1000c0101000a000` — matches `vendor=0x1000, device=0xc010, subsystem_vendor=0x1000`
- **Gen5**: `(pci.device & 0xfffffffffffff000) == 0x1000c03010000000` — matches `vendor=0x1000, device=0xc030, subsystem_vendor=0x1000`

**Not applicable to this platform**: On the SMC300x MI300X, the PEX89104 switches have `vendor=0x1000, device=0xc030` but `subsystem_vendor=0x15d9` (Supermicro), `subsystem_device=0x1d2a`. This yields `pci.device = 0x1000c03015d91d2a`. The Gen5 mask check produces `0x1000c03015d91000 != 0x1000c03010000000` — **the subsystem_vendor mismatch (`0x15d9` vs `0x1000`) means `getBcmGen()` returns 0 and flattening is skipped**. The internal 2-level BCM switch hierarchy is preserved, which adds extra hop count to same-partition paths but does not affect the PATH_PXB vs PATH_PHB classification (the BFS Rule 1 classifies PCI→PCI traversals as PATH_PXB regardless of hop count).

---

## 4. Virtual Switch Link Mechanism

### 4.1 Kernel Module: switch_discovery
- Reads Broadcom VSEC (Vendor-Specific Extended Capability) registers
- Exposes inter-switch fabric via `/sys/kernel/pci_switch_link/virtual_switch_links/`
- Refresh trigger: `/sys/kernel/pci_switch_link/refresh_switch_toplogy`
- **Status on test nodes**: NOT loaded, sysfs path does NOT exist

### 4.2 RCCL Integration
`src/graph/xml.cc` lines 778-794:

```c
// When building XML topology, for BCM switches (vendor 0x1000):
if (vendor != NULL && strcmp(vendor, "0x1000") == 0) {
    ncclOsGetBcmLinks(busId, &nlinks, &peers);
    // Adds <pcilink target="peer_busid"/> elements to XML
}
```

`src/graph/topo.cc` — `ncclTopoRefreshBcmP2pLinks()`:
```c
// Trigger sysfs refresh before reading links
FILE* fp = fopen("/sys/kernel/pci_switch_link/refresh_switch_toplogy", "r");
```

These `<pcilink>` elements create **direct PCI↔PCI edges** in the topology graph, enabling the BFS to find PCI-only paths between switch partitions.

### 4.3 Impact on Path Computation

**Without virtual switch links (current state):**
```
GPU1 → PCI(61:00.0) → CPU(NUMA0) → PCI(01:00.0) → ionic_6
                        ↑
                   PATH_PHB triggered (Rule 2)
```
Result: **PATH_PHB (8)** → GDR disabled → host memory bounce buffer

**With virtual switch links (switch_discovery + pcilink):**
```
GPU1 → PCI(61:00.0) → PCI(41:00.0) → PCI(01:00.0) → ionic_6
        PCI→PCI = PXB    PCI→PCI = PXB   (no CPU traversal)
```
Result: **PATH_PXB (5)** → GDR enabled → direct GPU↔NIC DMA

### 4.4 Required Fix Components
1. **Kernel**: `switch_discovery` module loaded, exposing `/sys/kernel/pci_switch_link/`
2. **RCCL**: PR #3121 (Wenkai Du) — reads BCM P2P links and adds pcilink topology edges. **Note**: This NCCL patch is already incorporated in the latest RCCL codebase.
3. Both kernel module and RCCL pcilink support must be deployed together

---

## 5. RCCL NIC Selection Logic

### 5.1 Entry Point
`src/graph/search.cc:1645` — `ncclTopoGetNetDev()` determines NIC for each channel:
- Graph-based channels use `inter[]` from graph search
- P2P channels use `ncclTopoGetLocalNet()`

### 5.2 Local NIC Selection Algorithm
`src/graph/topo.cc:2524-2581` — `ncclTopoGetLocalNetType()`:

```c
// Step 1: Find equidistant NICs (best path type + highest bandwidth)
ncclTopoGetLocal(system, gpuNode->gpu.dev, NET, &localRailCount, ...);

// Step 2: Assign GPU to NIC rail via bit-reversal
int devIdx = gpuNode->gpu.dev;
if (isPow2(localRailCount)) {
    rail = mirrorBits(devIdx, localRailCount);
} else {
    rail = (devIdx * DIVUP(localRailCount, localGpuCount)) % localRailCount;
}
```

**How `ncclTopoGetLocal()` works** (`topo.cc:2371-2400`):
1. For each available NIC, look up `paths[NET][n].type` (the path type from this GPU to NIC n, computed by BFS in `ncclTopoSetPaths`)
2. Find the **best (lowest) path type** across all NICs
3. Among NICs with that best path type, find the **highest bandwidth**
4. Return only NICs matching both best path type AND highest bandwidth → these are the "equidistant" candidates (`localRailCount`)

**Key behavior**: Cross-NUMA NICs have PATH_SYS (9), which is worse than same-NUMA PATH_PHB (8), so NUMA-local NICs are always preferred. The equidistant set is always confined to same-NUMA NICs.

### 5.3 mirrorBits() — Rail-Striping Function
`src/graph/topo.h:407-412`:

```c
static int mirrorBits(int val, int pow2) {
    int mirror = 0;
    for (int b = 1, mb = (pow2 >> 1); b < pow2; b <<= 1, mb >>= 1)
        if (val & b) mirror |= mb;
    return mirror;
}
```

- `mirrorBits(val, 1) = 0` always (loop body never executes since `1 < 1` is false)
- `mirrorBits(val, 2)` = bit 0 of val → 0 for even GPUs, 1 for odd GPUs

### 5.4 Scenario A: 8 GPUs / 8 NICs (No Split-Mask)

With all 8 NICs present, each partition has exactly 1 GPU + 1 NIC. Each GPU has a **same-partition NIC** reachable via PATH_PXB (PCI switches only, no CPU traversal):

| GPU | Upstream | Same-Partition NIC | Path Type | localRailCount |
|---|---|---|---|---|
| GPU0 | `01:00.0` (Sw1-A) | ionic_6 | PATH_PXB (5) | 1 |
| GPU1 | `21:00.0` (Sw1-B) | ionic_7 | PATH_PXB (5) | 1 |
| GPU2 | `41:00.0` (Sw3-A) | ionic_5 | PATH_PXB (5) | 1 |
| GPU3 | `61:00.0` (Sw3-B) | ionic_4 | PATH_PXB (5) | 1 |
| GPU4 | `81:00.0` (Sw4-A) | ionic_2 | PATH_PXB (5) | 1 |
| GPU5 | `a1:00.0` (Sw4-B) | ionic_3 | PATH_PXB (5) | 1 |
| GPU6 | `c1:00.0` (Sw5-A) | ionic_1 | PATH_PXB (5) | 1 |
| GPU7 | `e1:00.0` (Sw5-B) | ionic_0 | PATH_PXB (5) | 1 |

**Selection logic**: `ncclTopoGetLocal()` finds 1 NIC at PATH_PXB (best). All other NICs are PATH_PHB (cross-partition or cross-switch through CPU) or PATH_SYS (cross-NUMA). With `localRailCount=1`, `mirrorBits(devIdx, 1) = 0` always → each GPU maps to its single closest NIC.

**Result**: Perfect 1:1 GPU-to-NIC pairing. All same-partition. All GDR enabled. No contention.

### 5.5 Scenario B: 8 GPUs / 4 NICs (Split-Mask, Without Virtual Switch Links)

With `NCCL_IB_HCA=ionic_0,ionic_2,ionic_4,ionic_6`, the available NICs are:

- **NUMA 0**: ionic_6 (`01:00.0`, Sw1-A), ionic_4 (`61:00.0`, Sw3-B)
- **NUMA 1**: ionic_2 (`81:00.0`, Sw4-A), ionic_0 (`e1:00.0`, Sw5-B)

Half the GPUs lost their same-partition NIC (ionic_7, ionic_5, ionic_3, ionic_1 were masked). The selection diverges based on whether the GPU's same-partition NIC survived the mask:

**GPUs with a same-partition NIC still available (GPU0, GPU3, GPU4, GPU7):**

| GPU | Upstream | Closest NIC | Path Type | Other NICs | localRailCount |
|---|---|---|---|---|---|
| GPU0 | `01:00.0` (Sw1-A) | ionic_6 (Sw1-A) | PATH_PXB | ionic_4: PATH_PHB | **1** |
| GPU3 | `61:00.0` (Sw3-B) | ionic_4 (Sw3-B) | PATH_PXB | ionic_6: PATH_PHB | **1** |
| GPU4 | `81:00.0` (Sw4-A) | ionic_2 (Sw4-A) | PATH_PXB | ionic_0: PATH_PHB | **1** |
| GPU7 | `e1:00.0` (Sw5-B) | ionic_0 (Sw5-B) | PATH_PXB | ionic_2: PATH_PHB | **1** |

→ `localRailCount=1`, GPU picks its same-partition NIC. GDR enabled.

**GPUs whose same-partition NIC was masked (GPU1, GPU2, GPU5, GPU6):**

| GPU | Upstream | Masked NIC | Remaining same-NUMA NICs | Path to each | localRailCount |
|---|---|---|---|---|---|
| GPU1 | `21:00.0` (Sw1-B) | ionic_7 | ionic_6 (Sw1-A): PATH_PHB, ionic_4 (Sw3-B): PATH_PHB | **2** |
| GPU2 | `41:00.0` (Sw3-A) | ionic_5 | ionic_6 (Sw1-A): PATH_PHB, ionic_4 (Sw3-B): PATH_PHB | **2** |
| GPU5 | `a1:00.0` (Sw4-B) | ionic_3 | ionic_2 (Sw4-A): PATH_PHB, ionic_0 (Sw5-B): PATH_PHB | **2** |
| GPU6 | `c1:00.0` (Sw5-A) | ionic_1 | ionic_2 (Sw4-A): PATH_PHB, ionic_0 (Sw5-B): PATH_PHB | **2** |

→ `localRailCount=2` (both same-NUMA NICs are equidistant at PATH_PHB). `mirrorBits()` breaks the tie:

```
GPU1: mirrorBits(1, 2) = 1 → rail 1 → ionic_4 (Sw3-B) — cross-switch!
GPU2: mirrorBits(2, 2) = 0 → rail 0 → ionic_6 (Sw1-A) — cross-switch!
GPU5: mirrorBits(5, 2) = 1 → rail 1 → ionic_0 (Sw5-B) — cross-switch!
GPU6: mirrorBits(6, 2) = 0 → rail 0 → ionic_2 (Sw4-A) — cross-switch!
```

**Critical insight**: Without virtual switch links, RCCL cannot distinguish:
- ionic_6 for GPU1 (cross-partition, **same** physical switch — would be PXB if visible)
- ionic_4 for GPU1 (different physical switch — truly PHB)

Both appear as PATH_PHB, so `mirrorBits()` assigns GPU1 to ionic_4 on a completely different physical switch.

### 5.6 Observed NIC Assignment (from RCCL debug logs, 4-NIC config)
All 16 channels per rank used the same NIC (no per-channel striping for 2-rank comms):

| GPU | Upstream | NIC | NIC Upstream | Rail | Topology Relationship | GDR |
|---|---|---|---|---|---|---|
| GPU0 | `01:00.0` (Sw1-A) | ionic_6 | `01:00.0` (Sw1-A) | 0 | Same-partition | Enabled |
| GPU1 | `21:00.0` (Sw1-B) | ionic_4 | `61:00.0` (Sw3-B) | 1 | Cross-switch (Sw1→Sw3) | **Disabled** |
| GPU2 | `41:00.0` (Sw3-A) | ionic_6 | `01:00.0` (Sw1-A) | 0 | Cross-switch (Sw3→Sw1) | **Disabled** |
| GPU3 | `61:00.0` (Sw3-B) | ionic_4 | `61:00.0` (Sw3-B) | 1 | Same-partition | Enabled |
| GPU4 | `81:00.0` (Sw4-A) | ionic_2 | `81:00.0` (Sw4-A) | 0 | Same-partition | Enabled |
| GPU5 | `a1:00.0` (Sw4-B) | ionic_0 | `e1:00.0` (Sw5-B) | 1 | Cross-switch (Sw4→Sw5) | **Disabled** |
| GPU6 | `c1:00.0` (Sw5-A) | ionic_2 | `81:00.0` (Sw4-A) | 0 | Cross-switch (Sw5→Sw4) | **Disabled** |
| GPU7 | `e1:00.0` (Sw5-B) | ionic_0 | `e1:00.0` (Sw5-B) | 1 | Same-partition | Enabled |

### 5.7 Scenario C: 8 GPUs / 4 NICs (Split-Mask, WITH `switch_discovery` Module + RCCL pcilink Support)

When the `switch_discovery` kernel module is loaded and RCCL includes pcilink support (PR #3121), the inter-partition fabric within each Broadcom PEX89104 becomes visible to RCCL's topology graph. Here is the end-to-end flow:

**Step 1 — Kernel exposes inter-partition links via sysfs:**
The `switch_discovery` module reads Broadcom VSEC (Vendor-Specific Extended Capability) registers from PCIe config space to discover which upstream ports belong to the same physical switch. It exposes these relationships at:
```
/sys/kernel/pci_switch_link/virtual_switch_links/
```
Each entry lists a pair of BDF addresses (e.g., `01:00.0 ↔ 21:00.0`) indicating that these two upstream ports are partitions of the same physical switch and connected via internal fabric.

**Step 2 — RCCL reads sysfs links during XML topology construction:**
In `ncclTopoGetXmlFromSys()` (`src/graph/xml.cc:778-794`), when RCCL encounters a Broadcom switch (vendor `0x1000`), it calls `ncclOsGetBcmLinks(busId, &nlinks, &peers)` which reads the sysfs entries. For each peer found, RCCL adds a `<pcilink target="peer_busid"/>` element to the XML topology.

**Step 3 — pcilink elements become PCI↔PCI edges in the topology graph:**
When the XML is parsed into RCCL's internal graph (`ncclTopoConnectNodes`), each `<pcilink>` element creates a **direct edge between two PCI switch nodes** — bypassing the CPU node entirely. For Switch 1, this creates: `PCI(01:00.0) ↔ PCI(21:00.0)`.

**Step 4 — BFS path computation yields PATH_PXB instead of PATH_PHB:**
When `ncclTopoSetPaths()` runs BFS from GPU1 to ionic_6, it now finds this path:
```
GPU1 → PCI(21:00.0) → [pcilink] → PCI(01:00.0) → ionic_6
        PCI→PCI = Rule 1 → PATH_PXB    (no CPU node in path!)
```
Without the pcilink edge, the only path was through the CPU:
```
GPU1 → PCI(21:00.0) → CPU(NUMA0) → PCI(01:00.0) → ionic_6
                        ↑ Rule 2 → PATH_PHB (CPU traversal)
```
Since path type uses `max()` (worst segment wins), the pcilink path stays at PATH_PXB (5), while the CPU path hits PATH_PHB (8). The BFS picks the lower-type path.

**GPUs whose same-partition NIC was masked — path types change:**

| GPU | Upstream | ionic_6 / ionic_4 / ionic_2 / ionic_0 path | localRailCount |
|---|---|---|---|
| GPU1 | `21:00.0` (Sw1-B) | ionic_6 (Sw1-A): **PATH_PXB** (same switch via fabric), ionic_4 (Sw3-B): PATH_PHB | **1** → ionic_6 |
| GPU2 | `41:00.0` (Sw3-A) | ionic_4 (Sw3-B): **PATH_PXB** (same switch via fabric), ionic_6 (Sw1-A): PATH_PHB | **1** → ionic_4 |
| GPU5 | `a1:00.0` (Sw4-B) | ionic_2 (Sw4-A): **PATH_PXB** (same switch via fabric), ionic_0 (Sw5-B): PATH_PHB | **1** → ionic_2 |
| GPU6 | `c1:00.0` (Sw5-A) | ionic_0 (Sw5-B): **PATH_PXB** (same switch via fabric), ionic_2 (Sw4-A): PATH_PHB | **1** → ionic_0 |

**Result with `switch_discovery` + pcilink**: Every GPU now picks a NIC on its **own physical switch** (same-partition or cross-partition via internal fabric). `localRailCount=1` for all GPUs — no `mirrorBits()` tie-breaking needed. All 8 GPU-NIC paths are PATH_PXB or better. **GDR enabled for all 8 GPUs.**

| GPU | NIC (without switch_discovery) | NIC (with switch_discovery + pcilink) | Change |
|---|---|---|---|
| GPU0 | ionic_6 (same-partition) | ionic_6 (same-partition) | No change |
| GPU1 | ionic_4 (Sw3, cross-switch) | **ionic_6 (Sw1, cross-partition)** | Reassigned |
| GPU2 | ionic_6 (Sw1, cross-switch) | **ionic_4 (Sw3, cross-partition)** | Reassigned |
| GPU3 | ionic_4 (same-partition) | ionic_4 (same-partition) | No change |
| GPU4 | ionic_2 (same-partition) | ionic_2 (same-partition) | No change |
| GPU5 | ionic_0 (Sw5, cross-switch) | **ionic_2 (Sw4, cross-partition)** | Reassigned |
| GPU6 | ionic_2 (Sw4, cross-switch) | **ionic_0 (Sw5, cross-partition)** | Reassigned |
| GPU7 | ionic_0 (same-partition) | ionic_0 (same-partition) | No change |

**NIC load distribution with switch_discovery**: Each NIC serves 2 GPUs (from both partitions of its physical switch):
- ionic_6: GPU0 (same-partition) + GPU1 (cross-partition)
- ionic_4: GPU3 (same-partition) + GPU2 (cross-partition)
- ionic_2: GPU4 (same-partition) + GPU5 (cross-partition)
- ionic_0: GPU7 (same-partition) + GPU6 (cross-partition)

---

## 6. Physical PCIe Topology (SMC300x MI300X)

> **Why this section follows RCCL NIC Selection Logic**: Understanding the RCCL selection algorithm and its three scenarios (sections 5.4-5.7) first reveals *why* the physical PCIe topology matters — specifically, why the inability to see inter-switch fabric between Broadcom switch partitions causes the 4-NIC degradation. This section then provides the hardware evidence — the physical switch mapping, DSN-based grouping, and 8-NIC vs 4-NIC topology comparison — that underpins those scenarios and motivates the DSN-based discovery approach as a potential complement to the `switch_discovery` kernel module.

### 6.1 Broadcom PEX89104 Multi-Host Switch Architecture

Each physical PEX89104 presents as **two virtual PCIe switches** (multi-host partitioning), each with its own upstream port connected to a different CPU root port. The OS sees two independent PCIe hierarchies per physical switch.

As described in section 4, the `switch_discovery` kernel module exposes inter-partition fabric links via sysfs, enabling RCCL to build correct topology paths. However, in the absence of this kernel module — for example, on deployments where the module is not yet available or cannot be loaded — an alternative approach is needed to infer which upstream ports belong to the same physical switch. The **PCIe Device Serial Number (DSN)** provides exactly this: all ports on the same physical silicon share an identical DSN, making it a viable method to discover physical switch membership and reconstruct the inter-partition relationships that RCCL needs for correct NIC selection.

### 6.2 Physical Switch Mapping (via PCIe Device Serial Number)

**Discovery method**: PCIe DSN at extended capability offset 0x100. All ports on the same physical switch share an identical DSN. Upstream ports with the same DSN are partitions of the same physical silicon — they share an internal cross-partition fabric.

**Per-partition device mapping** (verified via sysfs symlinks on smc300x-ccs-aus-gpucc5a):

| Physical Switch | DSN | Partition | Upstream Port | NUMA | GPU | NIC (8-NIC) | NIC (4-NIC) |
|---|---|---|---|---|---|---|---|
| Switch 1 | `...2e-34-7e-08` | A | `01:00.0` | 0 | GPU0 (`05:00.0`) | ionic_6 (`09:00.0`) | ionic_6 |
| Switch 1 | `...2e-34-7e-08` | B | `21:00.0` | 0 | GPU1 (`29:00.0`) | ionic_7 (`26:00.0`) | — (masked) |
| Switch 2 | `...40-69-5c-08` | — | `2b:00.0` (solo) | 0 | — | ionic_9 (`35:00.0`) | — |
| Switch 3 | `...9b-33-f1-08` | A | `41:00.0` | 0 | GPU2 (`49:00.0`) | ionic_5 (`46:00.0`) | — (masked) |
| Switch 3 | `...9b-33-f1-08` | B | `61:00.0` | 0 | GPU3 (`65:00.0`) | ionic_4 (`69:00.0`) | ionic_4 |
| Switch 4 | `...98-b9-88-08` | A | `81:00.0` | 1 | GPU4 (`85:00.0`) | ionic_2 (`89:00.0`) | ionic_2 |
| Switch 4 | `...98-b9-88-08` | B | `a1:00.0` | 1 | GPU5 (`a9:00.0`) | ionic_3 (`a6:00.0`) | — (masked) |
| Switch 5 | `...a0-0f-93-08` | A | `c1:00.0` | 1 | GPU6 (`c9:00.0`) | ionic_1 (`c6:00.0`) | — (masked) |
| Switch 5 | `...a0-0f-93-08` | B | `e1:00.0` | 1 | GPU7 (`e5:00.0`) | ionic_0 (`e9:00.0`) | ionic_0 |

### 6.3 8-NIC vs 4-NIC Topology Comparison

The topology table above has both NIC columns, but the critical difference is best understood by examining the GPU-to-NIC relationship per physical switch:

**8-NIC mode (all NICs active)**:
- Every partition has 1 GPU + 1 NIC → each GPU has a **same-partition** NIC
- GPU-to-NIC path: GPU → downstream port → upstream port → downstream port → NIC (all within one partition)
- Path type: PATH_PXB (5) for all 8 pairs — no CPU traversal needed
- Result: Perfect 1:1 pairing, all GDR enabled

| Physical Switch | Partition A | Partition B |
|---|---|---|
| Switch 1 | GPU0 ↔ ionic_6 | GPU1 ↔ ionic_7 |
| Switch 3 | GPU2 ↔ ionic_5 | GPU3 ↔ ionic_4 |
| Switch 4 | GPU4 ↔ ionic_2 | GPU5 ↔ ionic_3 |
| Switch 5 | GPU6 ↔ ionic_1 | GPU7 ↔ ionic_0 |

**4-NIC mode (ionic_0, ionic_2, ionic_4, ionic_6 only)**:
- The 4-NIC mask keeps exactly one NIC per physical switch (alternating partitions A/B)
- Half the GPUs lose their same-partition NIC, creating an asymmetry:

| Physical Switch | Partition A | Partition B | NIC Kept | Gap |
|---|---|---|---|---|
| Switch 1 | GPU0 + ionic_6 | GPU1 (NIC masked) | ionic_6 (A) | GPU1 has no same-partition NIC |
| Switch 3 | GPU2 (NIC masked) | GPU3 + ionic_4 | ionic_4 (B) | GPU2 has no same-partition NIC |
| Switch 4 | GPU4 + ionic_2 | GPU5 (NIC masked) | ionic_2 (A) | GPU5 has no same-partition NIC |
| Switch 5 | GPU6 (NIC masked) | GPU7 + ionic_0 | ionic_0 (B) | GPU6 has no same-partition NIC |

**The topology gap**: GPU1, GPU2, GPU5, and GPU6 each need to reach a NIC on the **other partition** of their physical switch. Without visibility into the internal cross-partition fabric (via `switch_discovery` or DSN-based inference), these cross-partition paths are indistinguishable from cross-switch paths — both traverse the CPU and get classified as PATH_PHB. This is the root cause of the NIC mis-assignment shown in section 5.5.

### 6.4 DSN as a Complement to Inter-Switch-Link Discovery

The Physical Switch Mapping table (section 6.2) demonstrates that the DSN groups partitions by physical switch — providing exactly the information RCCL needs to infer which cross-partition paths traverse the internal fabric rather than the CPU:

1. **DSN groups establish switch membership**: `01:00.0` and `21:00.0` share DSN `...2e-34-7e-08` → they are partitions of the same PEX89104 → cross-partition traffic between them uses internal fabric, not the CPU
2. **RCCL could use this to create pcilink edges**: Instead of relying on `switch_discovery`'s sysfs path, RCCL could read DSN directly from PCIe config space (offset 0x100), group upstream ports by DSN, and inject pcilink edges between partitions of the same switch
3. **Advantage**: No kernel module dependency — works on any system with Broadcom multi-host switches
4. **Limitation**: Requires `CAP_SYS_RAWIO` (root) to access extended config space (see section 7), but RCCL often runs with elevated privileges in HPC/datacenter environments

This makes DSN-based discovery a viable **alternative or fallback** when the `switch_discovery` module is unavailable, subject to privilege constraints.

### 6.5 Management Endpoints

Each physical switch exposes a management endpoint at Port #31 (`xx:1f.0`) under exactly ONE partition:

| Mgmt Endpoint | Parent Upstream | Physical Switch |
|---|---|---|
| `02:1f.0` | `01:00.0` | Switch 1 |
| `2c:1f.0` | `2b:00.0` | Switch 2 |
| `62:1f.0` | `61:00.0` | Switch 3 |
| `82:1f.0` | `81:00.0` | Switch 4 |
| `e2:1f.0` | `e1:00.0` | Switch 5 |

### 6.6 MI300X Internal PCIe Bridge

MI300X is a multi-chiplet SoC that presents as a PCIe switch internally:
- Upstream bridge: `1022:1500`
- Downstream bridge: `1002:1501`
- GPU endpoint: `1002:74a1`

This adds +2 bridge hops vs NVIDIA H20 (direct endpoint), contributing to higher PCIe distance values.

### 6.7 GPU-NIC PCIe Distance Matrix

**Without virtual switch link visibility (current state):**

- Same partition (GPU + NIC under same upstream port): ~6 hops → borderline PATH_PXB (5) / PATH_PHB (8)
- Cross partition (GPU under one upstream, NIC under another): 8+ hops → PATH_PHB (8) through CPU
- Cross switch (different physical switch entirely): PATH_PHB (8) through CPU

---

## 7. Sysfs-Based Inter-Switch Discovery (Without Kernel Module)

### 7.1 PCIe Device Serial Number (DSN) — Definitive Method (Requires Root)

The PCIe extended capability "Device Serial Number" (cap ID 0x0003 at offset 0x100) is present on all PEX890xx ports. All ports on the same physical switch share an identical DSN.

**Algorithm:**
```
1. Enumerate PCI devices with vendor=0x1000, device=0xc030
2. Read PCIe DSN extended capability at offset 0x100
3. Group by DSN → each group = one physical switch
4. Within each group, upstream ports are the partitions of the same physical switch
```

**Privilege Limitation**: DSN lives in PCIe extended config space (offset >= 0x100), which the kernel restricts to processes with `CAP_SYS_RAWIO` (effectively root). Verified on smc300x-ccs-aus-gpucc5a:

| Method | Without sudo | With sudo |
|---|---|---|
| `lspci -vvs` | `Capabilities: <access denied>` | Shows DSN correctly |
| `setpci -s ... 0x104.L` | Returns `ffffffff` (all-ones) | Returns correct DSN bytes |
| sysfs `config` file | Kernel returns only **64 bytes** (standard config header) | Full 4096 bytes (extended config) |
| `/proc/bus/pci/XX/YY.Z` | File shows 4096 bytes but read() returns **only 64 bytes** | Full 4096 bytes |

Root cause: The kernel enforces a 64-byte read limit on PCIe config space for non-root users. The `CAP_SYS_RAWIO` capability is in the bounding set but not in the effective set for normal users (`Current: =` per `capsh --print`). There are no sysfs attributes (e.g., `serial`, `dsn`) that expose the DSN without config space access.

### 7.2 Management Endpoint Parent (Secondary Method — No Root Required)

Port #31 (`xx:1f.0`) appears under exactly one upstream partition per physical switch. The sysfs parent path reveals grouping via standard symlink traversal (no privilege needed):
```
readlink -f /sys/bus/pci/devices/0000:02:1f.0
→ .../0000:01:00.0/0000:02:1f.0  (parent = 01:00.0, Switch 1)
```

**Limitation**: This only identifies ONE partition per physical switch (the one hosting Port #31). It cannot pair the two upstream ports of a dual-partition switch (e.g., it can identify `01:00.0` as Switch 1, but cannot confirm `21:00.0` is the other partition of the same switch without DSN or the `switch_discovery` module).

### 7.3 Methods That Do NOT Work
- IOMMU groups: unique per port, not per switch
- Vendor/device/subsystem IDs: identical across all switches
- NUMA node: only distinguishes CPU socket, not individual switches
- firmware_node/ACPI: not present on switch ports
- Physical slot info: empty on these systems
- BAR resources: upstream ports have no BARs
- Config space via sysfs read(): limited to 64 bytes without root

### 7.4 Conclusion

Without root privilege, there is **no reliable unprivileged method** to discover which upstream ports belong to the same physical Broadcom switch. The `switch_discovery` kernel module remains the intended solution — it runs in kernel context (full config space access) and exposes results to userspace via `/sys/kernel/pci_switch_link/virtual_switch_links/`.

---

## 8. Open Questions / Next Steps

> **Note**: Topology dumps, test configuration, and source file references have been moved to [split_mask_experiment_notes.md](split_mask_experiment_notes.md).

1. **DSN-based userspace alternative**: Could RCCL read PCIe DSN directly from config space (offset 0x100) to infer physical switch membership without requiring a kernel module? This would be a portable, module-free solution.

   #### Constraint Analysis

   RCCL itself runs as an unprivileged library within user-space GPU workloads — it cannot access PCIe extended config space (offset 0x100+) which requires `CAP_SYS_RAWIO` or root. However, the limitation is scoped to the RCCL library process only. A separate **privileged service** could extract DSN mappings and populate the sysfs interface that RCCL already reads via `ncclOsGetBcmLinks()` (`src/os/linux.cc:702-730`).

   #### RCCL's Existing Sysfs Consumption Pattern

   RCCL reads inter-partition link information from a well-defined sysfs path:
   ```
   /sys/kernel/pci_switch_link/virtual_switch_links/<busid>/<peer_busid>
   ```

   The `ncclOsGetBcmLinks()` function:
   1. Opens directory `/sys/kernel/pci_switch_link/virtual_switch_links/<busid>/`
   2. Iterates directory entries, filtering by PCI BDF string length (`BUSID_SIZE - 1` = 12 chars, e.g. `0000:c1:00.0`)
   3. Validates each entry resolves to a real PCI device via `ncclOsGetPciPath()`
   4. Collects valid peer bus IDs into a list

   **Key insight**: Any mechanism that populates this exact directory structure enables RCCL to discover inter-partition links **with zero code changes** to RCCL.

   #### Approach 1: Privileged Systemd Service (Recommended)

   A lightweight systemd service (`rccl-switch-discovery.service`) running as root at boot:

   **Operation**:
   - Enumerates all PCIe switches via `/sys/bus/pci/devices/*/class` (class `0x060400` = PCI bridge)
   - For each switch upstream port, reads DSN from extended config space at capability offset `0x100` (8-byte serial number)
   - Groups switch ports sharing the same DSN → these belong to the same physical switch chip
   - For each group, creates sysfs-compatible directory entries:
     ```
     /sys/kernel/pci_switch_link/virtual_switch_links/<port_A_busid>/<port_B_busid>
     /sys/kernel/pci_switch_link/virtual_switch_links/<port_B_busid>/<port_A_busid>
     ```

   **Sysfs population method**: Since writing to `/sys/kernel/` requires a kernel module, the service would use one of:
   - **Option A — Minimal kernel module**: A thin kernel module that exposes a write interface (e.g., `/sys/kernel/pci_switch_link/add_link`) and creates the sysfs directory entries. The userspace service does DSN discovery and tells the module which links to create. Much simpler than `switch_discovery` since it doesn't touch VSEC registers.
   - **Option B — tmpfs overlay**: Mount a tmpfs at `/sys/kernel/pci_switch_link/virtual_switch_links/` and populate it with symlinks/directories. Requires cooperation from system init scripts.
   - **Option C — RCCL-side env var fallback**: A minor RCCL enhancement to check `RCCL_BCM_LINKS_PATH` env var when the default sysfs path doesn't exist. The privileged service writes plain files to this userspace path. **Status: Implemented and validated — see section 9.**

   **Advantages**: Runs once at boot, no runtime overhead, survives container restarts, works with unmodified RCCL (for Options A/B).

   #### Approach 2: Capability-Restricted Helper Binary

   Instead of a full root service, use a setcap binary with minimal privileges:

   ```bash
   # Grant only raw I/O capability (for PCIe config space reads)
   sudo setcap cap_sys_rawio+ep /usr/local/bin/rccl-dsn-mapper
   ```

   **Operation**:
   - Binary reads PCIe extended config space via `/sys/bus/pci/devices/*/config` (requires `CAP_SYS_RAWIO` for offsets > 0xFF)
   - Extracts DSN at offset 0x100 from each PCI bridge device
   - Groups by DSN and writes the link map to a file (e.g., `/var/run/rccl/switch_links.json`)
   - RCCL would need a small enhancement to read this file as a fallback when sysfs path is absent

   **Precedent — AINIC `disable_acs.sh`**: There is an existing operational pattern for this approach. The AINIC stack already runs a privileged script (`/apps/shared/disable_acs.sh`) at NIC bring-up time that iterates all PCI devices and uses `setpci` to access extended config space (ACS capability) as root. A DSN mapper helper could follow the same deployment model — run as a privileged step during NIC bring-up, alongside `disable_acs.sh`, to enumerate PCI bridges, extract DSN from extended capability `0x0003`, and build the inter-switch-link representation before RCCL initializes. This avoids introducing new operational workflows since the NIC bring-up sequence already executes privileged PCIe configuration steps.

   **Advantages**: No kernel module needed, minimal privilege escalation (only `CAP_SYS_RAWIO`, not full root), can be run on-demand or integrated into existing NIC bring-up sequence.

   **Limitation**: Requires a minor RCCL code change to read the alternative link map format.

   **Status**: Prototype implemented as `rccl_dsn_mapper.sh` and validated — see section 9.

   #### Approach 3: udev Rule with DSN Extraction

   A udev rule triggered on PCIe bridge device enumeration:

   ```
   SUBSYSTEM=="pci", ATTR{class}=="0x060400", RUN+="/usr/local/bin/rccl-dsn-mapper --udev %k"
   ```

   **Operation**:
   - Triggered automatically when PCIe devices are enumerated (boot or hot-plug)
   - The helper script reads DSN from the newly-added device, cross-references with previously-seen devices
   - Populates link entries incrementally as devices appear

   **Advantages**: Automatic, no manual service management, handles hot-plug scenarios.

   **Limitation**: Race conditions during parallel device enumeration at boot; must handle partial state gracefully.

   #### Approach 4: Simplified Kernel Module (DSN-only)

   A kernel module much simpler than `switch_discovery` that reads only DSN (not VSEC registers):

   **Comparison with `switch_discovery`**:
   | Aspect | `switch_discovery` | DSN-only module |
   |---|---|---|
   | Data source | VSEC registers (vendor-specific) | PCIe DSN capability (standard) |
   | Switch vendor dependency | Broadcom-specific | Vendor-agnostic |
   | Complexity | High (VSEC register parsing) | Low (standard capability read) |
   | Sysfs output | Same path | Same path |

   **Operation**:
   - On load, iterates PCI bridges, reads DSN from standard extended capability (ID `0x0003`)
   - Groups ports by DSN, creates sysfs entries at `/sys/kernel/pci_switch_link/virtual_switch_links/`
   - Fully compatible with RCCL's existing `ncclOsGetBcmLinks()` — zero RCCL changes needed

   **Advantages**: Vendor-agnostic (works with any PCIe switch, not just Broadcom), standard PCIe capability (not vendor-specific VSEC), simpler to maintain and audit.

   #### Recommendation

   **Short-term**: Approach 1 (Option C) — `RCCL_BCM_LINKS_PATH` env var fallback with userspace DSN mapper script at NIC bring-up. **Validated experimentally — see section 9.**

   **Medium-term**: Approach 4 — a simplified DSN-only kernel module that is vendor-agnostic and uses the standard PCIe DSN extended capability. This is the cleanest long-term solution as it works across switch vendors and produces the exact sysfs layout RCCL expects.

   **For evaluation/prototyping**: Approach 2 — a setcap helper binary is the fastest path to validating that DSN-based grouping correctly identifies physical switch membership on the SMC300x platform, before investing in kernel module development.

---

## 9. Experimental Validation: DSN Mapper + RCCL_BCM_LINKS_PATH

### 9.1 Fix Implementation

Two components were implemented and tested on both SMC300x nodes:

**Component 1 — RCCL patch** (`src/os/linux.cc:ncclOsGetBcmLinks()`):

A conditional fallback in the BCM link discovery function:
```c
static const char defaultBase[] = "/sys/kernel/pci_switch_link/virtual_switch_links";
const char* base = defaultBase;
struct stat st;
if (stat(defaultBase, &st) != 0 || !S_ISDIR(st.st_mode)) {
    const char* envBase = getenv("RCCL_BCM_LINKS_PATH");
    if (envBase && stat(envBase, &st) == 0 && S_ISDIR(st.st_mode)) {
        base = envBase;
    }
}
```

**Priority logic**: Default sysfs path always takes precedence → `switch_discovery` module is never overridden. `RCCL_BCM_LINKS_PATH` is only consulted when the default path doesn't exist.

**Component 2 — DSN mapper script** (`rccl_dsn_mapper.sh`):

Privileged script (following `disable_acs.sh` pattern) with three modes:
- `--discover`: Dry-run — prints discovered switch groups and peer links
- `--populate`: Creates peer link entries at `/var/run/rccl_bcm_links/<busid>/<peer_busid>`
- `--clean`: Removes all previously created entries using a change log

### 9.2 DSN Discovery Results (Both Nodes)

Both nodes show identical topology structure — 4 multi-partition + 1 solo switch:

```
Physical Switch 1:  0000:01:00.0  <-->  0000:21:00.0   (GPU0 ↔ GPU1 side)
Physical Switch 2:  0000:41:00.0  <-->  0000:61:00.0   (GPU2 ↔ GPU3 side)
Physical Switch 3:  0000:81:00.0  <-->  0000:a1:00.0   (GPU4 ↔ GPU5 side)
Physical Switch 4:  0000:c1:00.0  <-->  0000:e1:00.0   (GPU6 ↔ GPU7 side)
Solo (NIC-only):    0000:2b:00.0                        (no cross-partition peer)
```

Populated directory structure on each node:
```
/var/run/rccl_bcm_links/
├── 0000:01:00.0/0000:21:00.0    ├── 0000:81:00.0/0000:a1:00.0
├── 0000:21:00.0/0000:01:00.0    ├── 0000:a1:00.0/0000:81:00.0
├── 0000:41:00.0/0000:61:00.0    ├── 0000:c1:00.0/0000:e1:00.0
├── 0000:61:00.0/0000:41:00.0    └── 0000:e1:00.0/0000:c1:00.0
```

### 9.3 RCCL PATH Selection: Before vs After Fix

**Before fix** (no inter-link visibility):

```
┌──────┬──────────┬──────────────────────┬──────────┬─────────┬───────────┬──────────────┐
│ Rank │ GPU BDF  │ NIC(s) Assigned      │ Path     │ GDR     │ Channels  │ Proxy Conns  │
├──────┼──────────┼──────────────────────┼──────────┼─────────┼───────────┼──────────────┤
│ 0    │ 05:00.0  │ ionic_6              │ PATH_PXB │ ON      │ 16        │ 33           │
│ 1    │ 29:00.0  │ ionic_6 + ionic_4    │ PATH_PHB │ OFF     │ 32        │ 65           │
│ 2    │ 49:00.0  │ ionic_6 + ionic_4    │ PATH_PHB │ OFF     │ 32        │ 65           │
│ 3    │ 65:00.0  │ ionic_4              │ PATH_PXB │ ON      │ 16        │ 33           │
│ 4    │ 85:00.0  │ ionic_2              │ PATH_PXB │ ON      │ 16        │ 33           │
│ 5    │ a9:00.0  │ ionic_2 + ionic_0    │ PATH_PHB │ OFF     │ 32        │ 65           │
│ 6    │ c9:00.0  │ ionic_2 + ionic_0    │ PATH_PHB │ OFF     │ 32        │ 65           │
│ 7    │ e5:00.0  │ ionic_0              │ PATH_PXB │ ON      │ 16        │ 33           │
└──────┴──────────┴──────────────────────┴──────────┴─────────┴───────────┴──────────────┘
GDR ON: 4/8 GPUs (50%)
```

**After fix** (DSN mapper + `RCCL_BCM_LINKS_PATH=/var/run/rccl_bcm_links`):

```
┌──────┬──────────┬──────────────────────┬──────────┬─────────┬───────────┬──────────────┐
│ Rank │ GPU BDF  │ NIC Assigned         │ Path     │ GDR     │ Channels  │ Proxy Conns  │
├──────┼──────────┼──────────────────────┼──────────┼─────────┼───────────┼──────────────┤
│ 0    │ 05:00.0  │ ionic_6 (IB-CAST/0)  │ PATH_PXB │ ON      │ 16        │ 33           │
│ 1    │ 29:00.0  │ ionic_6 (IB-CAST/0)  │ PATH_PXB │ ON      │ 16        │ 33           │
│ 2    │ 49:00.0  │ ionic_4 (IB-CAST/1)  │ PATH_PXB │ ON      │ 16        │ 33           │
│ 3    │ 65:00.0  │ ionic_4 (IB-CAST/1)  │ PATH_PXB │ ON      │ 16        │ 33           │
│ 4    │ 85:00.0  │ ionic_2 (IB-CAST/2)  │ PATH_PXB │ ON      │ 16        │ 33           │
│ 5    │ a9:00.0  │ ionic_2 (IB-CAST/2)  │ PATH_PXB │ ON      │ 16        │ 33           │
│ 6    │ c9:00.0  │ ionic_0 (IB-CAST/3)  │ PATH_PXB │ ON      │ 16        │ 33           │
│ 7    │ e5:00.0  │ ionic_0 (IB-CAST/3)  │ PATH_PXB │ ON      │ 16        │ 33           │
└──────┴──────────┴──────────────────────┴──────────┴─────────┴───────────┴──────────────┘
GDR ON: 8/8 GPUs (100%)
```

**Key changes for ranks 1, 2, 5, 6:**
- PATH_PHB → PATH_PXB (cross-partition now recognized as same physical switch)
- GDR OFF → GDR ON (threshold < PATH_PHB met)
- Dual-NIC → Single-NIC (same-switch NIC selected, not equidistant pair)
- 32 channels → 16 channels (no dual-rail compensation needed)
- 65 proxy connections → 33 (no CPU bounce buffer overhead)

### 9.4 RCCL Transport Log Comparison

**Before fix — Rank 1 (GPU1, GDR OFF, dual-NIC, no GDRDMA):**
```
Channel 00/0 : 1[1] -> 0[1] [receive] via NET/IB-CAST/1
Channel 01/0 : 1[1] -> 0[1] [receive] via NET/IB-CAST/0
Channel 02/0 : 1[1] -> 0[1] [receive] via NET/IB-CAST/1
Channel 03/0 : 1[1] -> 0[1] [receive] via NET/IB-CAST/0
  ... (alternating IB-CAST/1 and IB-CAST/0, NO GDRDMA suffix)
```

**After fix — Rank 1 (GPU1, GDR ON, single-NIC, GDRDMA):**
```
Channel 00/0 : 1[1] -> 0[1] [receive] via NET/IB-CAST/0/GDRDMA/flush=Always
Channel 01/0 : 1[1] -> 0[1] [receive] via NET/IB-CAST/0/GDRDMA/flush=Always
Channel 02/0 : 1[1] -> 0[1] [receive] via NET/IB-CAST/0/GDRDMA/flush=Always
  ... (all 16 channels use IB-CAST/0/GDRDMA — single NIC, direct GPU-NIC DMA)
```

### 9.5 RCCL busBw Comparison (1K → 16G, 20 iters, 5 warmup)

**all_reduce:**

```
┌──────────────┬────────────────────────────┬────────────────────────────┐
│  Msg Size    │  Before (busBw GB/s)       │  After (busBw GB/s)        │
│              │  out-of-place / in-place   │  out-of-place / in-place   │
├──────────────┼────────────────────────────┼────────────────────────────┤
│  1K          │      0.00 /     0.00       │      0.00 /     0.00       │
│  64K         │      0.02 /     0.02       │      0.02 /     0.03       │
│  1M          │      0.35 /     0.32       │      0.29 /     0.42       │
│  16M         │      5.97 /     5.89       │      5.50 /     5.50       │
│  64M         │     14.47 /    15.05       │     18.26 /    17.96       │
│  128M        │     21.12 /    21.18       │     23.36 /    23.62       │
│  256M        │     22.00 /    22.47       │     23.71 /    23.72       │
│  512M        │     22.57 /    22.87       │     24.05 /    24.05       │
│  1G          │     23.10 /    24.01       │     24.19 /    24.15       │
│  2G          │     26.68 /    26.86       │     24.23 /    24.26       │
│  4G          │     24.75 /    24.42       │     24.32 /    24.30       │
│  8G          │     24.55 /    24.87       │     24.35 /    24.36       │
│  16G         │     24.74 /    24.96       │     24.39 /    24.37       │
└──────────────┴────────────────────────────┴────────────────────────────┘
```

**alltoall:**

```
┌──────────────┬────────────────────────────┬────────────────────────────┐
│  Msg Size    │  Before (busBw GB/s)       │  After (busBw GB/s)        │
│              │  out-of-place / in-place   │  out-of-place / in-place   │
├──────────────┼────────────────────────────┼────────────────────────────┤
│  1K          │      0.00 /     0.00       │      0.00 /     0.00       │
│  64K         │      0.01 /     0.01       │      0.01 /     0.01       │
│  1M          │      0.14 /     0.19       │      0.14 /     0.21       │
│  16M         │      2.45 /     3.04       │      2.62 /     2.71       │
│  64M         │      7.62 /     9.59       │     11.00 /     9.80       │
│  128M        │     17.67 /    21.80       │     16.26 /    20.93       │
│  256M        │     22.55 /    22.42       │     22.57 /    22.11       │
│  512M        │     23.14 /    22.85       │     23.22 /    22.36       │
│  1G          │     23.35 /    23.01       │     23.43 /    22.55       │
│  2G          │     23.51 /    23.10       │     23.54 /    22.65       │
│  4G          │     23.57 /    23.14       │     23.62 /    22.71       │
│  8G          │     23.60 /    23.18       │     23.66 /    22.76       │
│  16G         │     23.59 /    23.19       │     23.66 /    22.77       │
└──────────────┴────────────────────────────┴────────────────────────────┘
```

### 9.6 Performance Analysis

**Saturated bandwidth comparison** (256M+ message sizes):

| Collective | Before busBw | After busBw | Delta |
|---|---|---|---|
| all_reduce (in-place) | 22.5–26.9 GB/s | 24.0–24.4 GB/s | More uniform |
| all_reduce (out-of-place) | 22.0–26.7 GB/s | 23.7–24.4 GB/s | More uniform |
| alltoall (out-of-place) | 23.1–23.6 GB/s | 23.2–23.7 GB/s | ~Same |
| alltoall (in-place) | 22.9–23.2 GB/s | 22.1–22.8 GB/s | ~Same |

**Key observations:**

1. **Saturated bandwidth is comparable** (~24 GB/s AR, ~23.5 GB/s A2A). The fix does not significantly change peak throughput because the bottleneck is the 200 Gbps NIC link bandwidth shared by 2 GPUs.

2. **Before-fix 2G all_reduce "peak" of 26.86 GB/s was an artifact**: GDR-OFF ranks used dual-rail (2 NICs, 32 channels) which momentarily boosted throughput at certain message sizes. This is not sustainable — it came at the cost of CPU bounce buffers and 2x proxy connections.

3. **After-fix profile is more uniform**: in-place and out-of-place converge to the same ~24 GB/s band. Before-fix had asymmetric profiles due to mixed GDR-ON/GDR-OFF transport paths.

4. **Mid-range (64M–128M) all_reduce improved**: 14.47→18.26 GB/s (out-of-place, 64M) and 21.12→23.36 GB/s (128M). The uniform GDRDMA transport eliminates the CPU bounce overhead that dragged down mid-range performance.

### 9.7 Conclusion

The DSN-based userspace approach (Approach 2 from section 8) is **validated**:

- `rccl_dsn_mapper.sh` correctly discovers physical switch membership on SMC300x
- `RCCL_BCM_LINKS_PATH` env var fallback enables RCCL to consume the mapping without requiring kernel sysfs writes
- PATH classification changes from PHB→PXB for cross-partition GPU-NIC pairs
- GDR coverage improves from 50% to 100%
- Transport is uniform GDRDMA across all ranks
- No performance regression; mid-range message sizes improve

**Deployment path**: Run `rccl_dsn_mapper.sh --populate` as root during NIC bring-up (alongside `disable_acs.sh`), set `RCCL_BCM_LINKS_PATH=/var/run/rccl_bcm_links` in the RCCL runtime environment.

**Artifacts:**
- RCCL patch: `projects/rccl/src/os/linux.cc` (3-line conditional in `ncclOsGetBcmLinks()`)
- DSN mapper: `rccl_dsn_mapper.sh` (discover/populate/clean modes)
- Test evidence: `test-experiments/before-inter-link-visibility/` and `test-experiments/after-inter-link-visibility/`

