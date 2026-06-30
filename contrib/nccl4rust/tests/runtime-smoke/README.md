# nccl4rust multi-GPU runtime smoke

This standalone package exercises the Rust host wrappers and the CUDA-Oxide
device boundary on two GPUs in one process. Each rank runs on its own host
thread and current CUDA device. The smoke:

1. initializes a two-rank communicator with `Communicator::init_rank`;
2. verifies a typed `f32` all-reduce;
3. creates and copies a public NCCL device-communicator image;
4. launches the CUDA-Oxide `nccl4rust_query_smoke` kernel from the linked
   device cubin and validates its world/LSA query results; and
5. explicitly destroys the device communicator and finalizes the host
   communicator.

Build the matching NCCL library, shim, and CUDA-Oxide cubin first. Then run on
a node with at least two `sm_90` GPUs:

```bash
export CUDA_HOME=/path/to/cuda
export NCCL_INCLUDE_DIR=/path/to/nccl/include
export NCCL_LIB_DIR=/path/to/nccl/lib
export LD_LIBRARY_PATH="$NCCL_LIB_DIR:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL4RUST_CUBIN=../cuda-oxide-smoke/nccl4rust_cuda_oxide_smoke_sm_90.cubin

cargo run --release --offline
```

On the local H100 Slurm partition, GPUs 0 and 3 are a direct NVLink pair and
the node has no configured GPU GRES. Allocate the whole node and select those
devices explicitly:

```bash
srun -p h100x8-cicd -N1 -n1 --exclusive -c16 --mem=64G -t 00:10:00 \
  bash -lc 'export CUDA_VISIBLE_DEVICES=0,3; \
    export LD_LIBRARY_PATH="$NCCL_LIB_DIR:$CUDA_HOME/lib64"; \
    timeout 180s cargo run --release --offline'
```

Build NCCL with `-DCMAKE_CUDA_ARCHITECTURES=90` for this node. A library that
contains only another architecture can fall back to embedded PTX, and PTX from
a toolkit newer than the installed driver may fail JIT compilation.

The package deliberately has its own empty workspace so GPU runtime linkage is
not imposed on the main host/device binding workspace.
