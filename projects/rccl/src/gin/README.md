# GIN rocSHMEM Plugins for RCCL

GPU-Initiated Networking (GIN) enables RCCL collectives to issue network
operations directly from GPU kernels, bypassing host-side proxy threads.
This directory contains the AMD-specific GIN backends built on top of
rocSHMEM:

| Backend | `NCCL_GIN_TYPE` | Description |
|---------|----------------|-------------|
| GDA (QueuePair) | 5 | GPU-initiated RDMA via IB QueuePairs posted from device code |
| Anvil SDMA | 6 | GPU-initiated DMA via the SDMA engine (Anvil) |

The upstream GIN framework also provides a host-side IB proxy backend
(`NCCL_GIN_TYPE=2`) that does not require rocSHMEM.

## Building

Building with `--rocshmem-gin` requires the
[rocm-systems](https://github.com/ROCm/rocm-systems) mono-repo, which
provides both `projects/rccl` and `projects/rocshmem` in a single tree.
`install.sh` auto-detects the sibling rocshmem project and builds it from
source via ExternalProject.  The device linker pipeline needs access to
intermediate build artifacts (per-arch `.bc` files under
`projects/rocshmem/build/bitcode/`) that are not part of the installed
package, so a pre-built rocSHMEM install is not sufficient.

RCCL provides two rocSHMEM integration modes (mutually exclusive):

```
cd projects/rccl
./install.sh --rocshmem-gin --amdgpu_targets=gfx950
```

**`--rocshmem-gin`** builds rocSHMEM from source for headers, device bitcode,
and host libraries.  `librocshmem.a` is **not** linked into `librccl.so`.
GIN device symbols (QueuePair) are resolved via per-arch bitcode injection
in the device linker pipeline.  GIN host plugin symbols are left unresolved
in `librccl.so` (`--allow-shlib-undefined`) and resolved at runtime from
the executable, which links `librocshmem.a` and exports symbols via
`-rdynamic`.

**`--rocshmem`** (separate feature) links `librocshmem.a` into `librccl.so`
for the alltoall\_wg offload path.  Does not enable GIN plugins.

## Running

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NCCL_GIN_ENABLE` | `1` | Enable/disable GIN (all backends) |
| `NCCL_GIN_TYPE` | (auto) | Force a specific backend: `5`=GDA, `6`=SDMA |
| `NCCL_GIN_ANVIL_SDMA_THRESHOLD` | `128` | Minimum message size (bytes) to use SDMA; smaller messages fall back to the IB proxy |
| `NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL` | `0` | Enable fused signal mode for SDMA (experimental) |
| `RCCL_GIN_ALLREDUCE_FORCE_ENABLE` | `0` | GIN AllReduce is used only for messages >= 256 MiB by default (smaller sizes use DDA). Set to `1` to also use GIN AllReduce for smaller messages (LSA one-shot / LSA two-shot). |
| `NCCL_CUMEM_ENABLE` | `0` | Required: GIN needs `hipMemCreate`-based allocations |
| `NCCL_DMABUF_ENABLE` | `0` | Recommended: enables dmabuf-based MR registration |
| `NCCL_P2P_DISABLE` | `0` | Set to `1` to force inter-GPU traffic over the network (useful for single-node GIN testing) |

### Example: AlltoAll with SDMA backend (8 GPUs, single node)

```sh
docker run -it --rm --shm-size 64G \
    --network host --device /dev/dri --device /dev/kfd \
    --device /dev/infiniband --ipc host \
    --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --privileged \
    rccl-gin \
    mpirun -n 8 -mca pml ob1 -mca btl ^openib \
        -x NCCL_GIN_ENABLE=1 \
        -x NCCL_GIN_TYPE=6 \
        -x NCCL_GIN_ANVIL_SDMA_THRESHOLD=128 \
        -x NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL=0 \
        -x NCCL_CUMEM_ENABLE=1 \
        -x NCCL_DMABUF_ENABLE=1 \
        -x NCCL_P2P_DISABLE=1 \
        -x NCCL_CROSS_NIC=1 \
        -x NCCL_IB_MERGE_NICS=0 \
        -x NCCL_MSCCL_ENABLE=0 \
        -x HSA_NO_SCRATCH_RECLAIM=1 \
        -x NCCL_DEBUG=INFO \
        -x NCCL_DEBUG_SUBSYS=INIT,NET \
        rccl-tests/alltoall_perf -b 128 -e 1024M -f 2 -g 1 -R 2 -D 3 -A 1
```

To use the GDA backend instead, set `NCCL_GIN_TYPE=5`.

### Platform requirements

GDA (`NCCL_GIN_TYPE=5`) requires a supported GDA NIC provider (mlx5, bnxt,
or ionic) with the corresponding user-space RDMA driver and firmware.
See the [rocSHMEM GDA NIC dependencies](https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/install.html#gda-nic-dependencies).

SDMA (`NCCL_GIN_TYPE=6`) requires an Anvil-capable GPU (MI300/MI350 class)
and `NCCL_CUMEM_ENABLE=1`.

## Known limitations

- **GDA on symmetric kernels**: the `__constant__` device memory
  (`constmem`, `logd_constants`) used by QueuePair is zero-initialized via
  stub definitions in `gin_rocshmem_constmem.hip`.  This is sufficient for
  SDMA (which does not read constmem) but GDA dispatch on symmetric kernels
  (e.g. `reduce_scatter_gin_*`) is not functional until a proper
  `hipMemcpyToSymbol` init path is added to librccl.so.
- **GDA on AlltoAllPivot**: works via the specialized kernel path (per-arch
  bitcode injection via `--rocshmem-bitcode`).

## Writing tests that use the GIN device API

Test executables that compile GIN device templates (e.g. `alltoall.cu`
with `ENABLE_ROCSHMEM_GIN`) must link `roc::rocshmem` and use
`-fgpu-rdc --hip-link -rdynamic`.  The `roc::rocshmem` target brings in
`hip::device` which provides `--offload-arch` flags automatically.
The `-rdynamic` flag exports host symbols so that `librccl.so` can resolve
GIN plugin functions from the executable at runtime.

## Source layout

| File | Role |
|------|------|
| `gin_host.cc` | GIN framework: backend selection, type negotiation |
| `gin_host_proxy.cc` | IB proxy backend (upstream, type 2) |
| `gin_plugin_rocshmem_gda.cc` | GDA plugin vtable (type 4) |
| `gin_plugin_anvil_sdma.cc` | SDMA Anvil plugin vtable (type 5) |
| `gin_rocshmem_gda_factory.cc` | QueuePair creation, MR registration, topology discovery |
| `gin_rocshmem_constmem.hip` | Stub `__constant__` definitions for device bitcode linking |
| `gin_anvil_ipc_table_host.cc` | IPC table management for SDMA |
| `gin_anvil_sdma_oss7_device.cc` | SDMA OSS7 device helpers |
