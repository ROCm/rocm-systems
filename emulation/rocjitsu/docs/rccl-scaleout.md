# Two-node RCCL NET smoke (unqualified control)

`configs/gfx1250_mi455x_kmd_2node_net.json` is a schematic system
manifest with one gfx1250 GPU on each of two nodes. Its network link records
intended reachability. The GPUs have distinct KFD identities and each node
projects only its own GPU into a separate rocJITsu daemon.

The same-host runner starts one rank per node and requests the loopback Socket
transport. With `RcclTest.*`, it checks all five rank tests against host
reference data and requires a data channel selecting `NET/Socket` in each rank
log before reporting success.
This is a control for multi-node launch and RCCL NET traffic. The manifest's
network link does not grant peer memory access or model a coherent fabric.

This gfx1250 daemon control has no passing rank logs or qualified SDK version.
With a RelWithDebInfo build and TheRock `10.1.0a20260822`, both ranks reached
AllReduce setup but faulted with SIGBUS in `ncclCudaHostCallocDebug` during
communicator initialization, before NET channel selection or numerical checks.
The same fault occurred with one daemon rank using the legacy gfx1250 config;
local mode and a gfx950 two-node NET control passed. A clean build of pre-PR
ancestor `6d9350d62f` reproduced the single-daemon SIGBUS, so this is a known
daemon prerequisite rather than evidence of a manifest regression. At that
tested revision, `AMDKFD_IOC_EXPORT_DMABUF` returned a daemon-local descriptor
number without transferring the descriptor to the client. This branch now
passes it through the RPC `shared_handle` and SCM_RIGHTS path. Bypassing the
bad close in the earlier run exposed a separate `hipHostMalloc` failure, so
the descriptor fix alone cannot qualify this recipe. Treat the command below
as a reproduction recipe until a passing gfx1250 NET rank run and its logs
are recorded.

## Build and run

Use a ROCm installation whose HIP compiler, runtime, and RCCL library support
gfx1250. Build the VFIO-enabled Release tree if following the QEMU guide, or
use another Release tree with the same rank target:

```bash
cmake --build build-vfio --target rocjitsu_bin hip_rccl_gfx1250_test_target

python3 scripts/run-rccl-scaleout.py \
  --rocjitsu build-vfio/tools/rocjitsu/rocjitsu \
  --config configs/gfx1250_mi455x_kmd_2node_net.json \
  --rank-binary build-vfio/tests/hip_rccl_gfx1250_test \
  --node node0 --node node1 \
  --output /tmp/rocjitsu-rccl-two-node-01 \
  --timeout 180 \
  --gtest-filter 'RcclTest.*' \
  --seed 79
```

The output directory must be new for each run. It contains `rank0.log` and
`rank1.log`, a shared RCCL communicator directory, and separate runtime
directories for the two daemons. The runner clears inherited GPU visibility
masks and GoogleTest controls so each rank's `--expected-devices=1` assertion
checks node-local HIP discovery and both ranks execute the same cases. It sets
`NCCL_NET=Socket`, selects loopback, and disables IB, P2P, and SHM. It bounds
the run and terminates both process groups on failure or interruption.

A passing result requires zero exit status and every requested GoogleTest case
passing on both ranks, plus `Channel ... via NET/Socket/` in both logs. Use
`--gtest-filter RcclTest.AllReduce` for a shorter first check. This runner
uses same-host loopback. The multi-host MPI and guest VFIO gates require
separate runs with their own transport and numerical evidence.
