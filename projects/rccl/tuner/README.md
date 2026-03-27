# RCCL CSV Tuner Configuration

This directory is installed to `${ROCM_PATH}/share/rccl/tuner/` when RCCL is built and installed. Any CSV files added here will be automatically copied to the install location.

## How to Use

### For Development (running from build directory)

Place your CSV config files in the `tuner/` directory within your build directory:

```bash
# From the build directory (e.g., build/release/)
cp rccl_tuner_gfx950.csv tuner/
./rccl-tests/all_reduce_perf  # Will auto-detect tuner/rccl_tuner_gfx950.csv
```

### For Production (after installing RCCL)

Place your CSV config files in the installed location:

```bash
sudo cp rccl_tuner_gfx950.csv /opt/rocm/share/rccl/tuner/
```

RCCL will automatically detect and load the config file at runtime.

## Auto-Discovery Order

The built-in CSV tuner searches for config files in this order:

1. `NCCL_TUNER_CONFIG_FILE` environment variable (if set)
2. `./tuner/rccl_tuner_<arch>.csv` (local, for development)
3. `./tuner/rccl_tuner.csv` (local, for development)
4. `${ROCM_PATH}/share/rccl/tuner/rccl_tuner_<arch>.csv` (GPU-specific, e.g., `rccl_tuner_gfx950.csv`)
5. `${ROCM_PATH}/share/rccl/tuner/rccl_tuner.csv` (generic fallback)

## CSV Format

Each line contains 8-10 comma-separated fields:

```
colltype,minbytes,maxbytes,algorithm,protocol,channels,nNodes,nRanks[,numPipeOps][,regBuff]
```

### Fields

| Field | Values | Description |
|-------|--------|-------------|
| colltype | `broadcast`, `reduce`, `allgather`, `reducescatter`, `allreduce` | Collective operation type |
| minbytes | integer | Minimum message size in bytes |
| maxbytes | integer | Maximum message size in bytes |
| algorithm | `tree`, `ring`, `collnet_direct`, `collnet_chain`, `nvls`, `nvls_tree`, `pat` | Algorithm to use |
| protocol | `ll`, `ll128`, `simple` | Protocol to use |
| channels | integer or `-1` | Number of channels (`-1` = RCCL default) |
| nNodes | integer or `-1` | Node count to match (`-1` = any) |
| nRanks | integer or `-1` | Rank count to match (`-1` = any) |
| numPipeOps | integer or `-1` | (Optional) Pipeline ops (`-1` = any) |
| regBuff | `0`, `1`, or `-1` | (Optional) Buffer registration (`-1` = any) |

### Example Config

```csv
# Tuning for 2-node gfx950 with 16 ranks
allreduce,0,65536,tree,ll,8,2,16,-1,-1
allreduce,65537,1048576,ring,ll128,16,2,16,-1,-1
allreduce,1048577,17179869184,ring,simple,64,2,16,-1,-1

# AllGather tuning
allgather,0,32768,ring,ll,4,-1,-1,-1,-1
```

## Disabling the Tuner

To disable the built-in CSV tuner even when a config file exists:

```bash
export NCCL_TUNER_PLUGIN=none
```
