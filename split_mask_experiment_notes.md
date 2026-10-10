# RCCL SPLIT_MASK Experiment Notes
# MI300X 8-GPU / 4-NIC per Node — BD Customer Investigation

**Date**: 2026-10-09
**Platform**: SMC300x MI300X (2 nodes: smc300x-ccs-aus-gpucc5a, smc300x-ccs-aus-gpuf273)
**Related**: [split_mask_pcie_topo_analysis_spec.md](split_mask_pcie_topo_analysis_spec.md)

---

## 1. Topology Dumps Collected

### 1.1 8-GPU / 8-NIC Configuration
File: `topo_8gpu_8nic.xml`
- All 8 GPUs and 8 NICs visible
- Each GPU+NIC pair under same BRCM switch partition → gdr=1
- NUMA 0: GPUs 0-3, NUMA 1: GPUs 4-7

### 1.2 8-GPU / 4-NIC Configuration (SPLIT_MASK)
File: `topo_8gpu_4nic_splitmask.xml`
- Only 4 NICs visible: ionic_0, ionic_2, ionic_4, ionic_6
- Filtered by `NCCL_IB_HCA=ionic_0,ionic_2,ionic_4,ionic_6`
- Shows rank 0's view (1 GPU per communicator per node)
- 3 of 4 NICs on different switch partitions from the GPU → cross-partition paths

---

## 2. Test Configuration

### Environment Variables (BD Experiment Flags)
```bash
NCCL_MAX_NCHANNELS=64
NCCL_TESTS_SPLIT_MASK=0x7
NCCL_NCHANNELS_PER_NET_PEER=16
NCCL_P2P_NET_CHUNKSIZE=262144
RCCL_P2P_SHIFT_SIZE=0
NCCL_IB_HCA=ionic_0,ionic_2,ionic_4,ionic_6   # 4-NIC filter
```

### Scripts
- `setup_env.sh`: Environment setup (RCCL_HOME, RCCL_TESTS, OMPI paths, node list)
- `run-rccl.sh`: mpirun wrapper with all RCCL env vars

### Run Command
```bash
bash run-rccl.sh all_reduce 1G 1G 1 1
```

### Compute Nodes

| Node | Role |
|---|---|
| `smc300x-ccs-aus-gpucc5a` | Node 1 (ranks 0-7) |
| `smc300x-ccs-aus-gpuf273` | Node 2 (ranks 8-15) |

---

## 3. Key Source Files Reference

| File | Key Functions | Purpose |
|---|---|---|
| `src/include/graph.h` | PATH_* defines | Path type constants |
| `src/graph/paths.cc` | `ncclTopoSetPaths()`, `ncclTopoComputePaths()` | BFS path computation, GDR/PXN decisions |
| `src/graph/topo.cc` | `ncclTopoGetLocalNetType()`, `ncclTopoGetLocal()`, `ncclTopoFlattenBcmSwitches()`, `ncclTopoRefreshBcmP2pLinks()` | NIC selection, BCM flattening, virtual link refresh |
| `src/graph/topo.h` | `mirrorBits()` | GPU-to-rail assignment |
| `src/graph/xml.cc` | `ncclTopoGetXmlFromSys()`, `ncclOsGetBcmLinks()` | XML topo building, BCM link reading |
| `src/graph/search.cc` | `ncclTopoGetNetDev()` | Per-channel NIC assignment |
| `src/init.cc:1928-1935` | Topo dump logic | NCCL_TOPO_DUMP_FILE handling |
| `rccl-tests/src/common.cu:2467-2494` | SPLIT_MASK parsing | MPI_Comm_split logic |
