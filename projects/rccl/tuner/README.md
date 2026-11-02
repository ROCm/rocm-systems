# RCCL CSV Tuner Configuration

Every CSV file in this directory is compiled into `librccl.so` at build time and used as the stock tuning
default. Nothing has to be staged on disk, so a cluster does not need the file on NFS or copied to every
node. The CSV text is embedded verbatim and parsed exactly as a file would be.

Adding or editing a CSV here and rebuilding is all that is needed to change the shipped defaults. The
generator is `cmake/GenerateEmbeddedTunerConfigs.cmake`; it writes `rccl_tuner_embedded_configs.h` into the
build tree's `include/` directory.

## How to Use

### Shipped defaults

Nothing to do. On a `gfx950` GPU, RCCL loads the embedded `rccl_tuner_gfx950.csv` automatically. Confirm
with:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING ./all_reduce_perf -b 1K -e 256M -f 2
# NCCL INFO Using built-in CSV tuner, config: <embedded>/rccl_tuner_gfx950.csv
```

### Overriding with your own file

A CSV on disk takes precedence over the embedded defaults. Either point at it directly:

```bash
export NCCL_TUNER_CONFIG_FILE=/path/to/my_tuning.csv
```

or drop it in a discovered location:

```bash
mkdir -p build/release/tuner && cp my_tuning.csv build/release/tuner/rccl_tuner_gfx950.csv  # development
sudo mkdir -p /opt/rocm/share/rccl/tuner && sudo cp my_tuning.csv /opt/rocm/share/rccl/tuner/rccl_tuner_gfx950.csv  # installed
```

## Auto-Discovery Order

The built-in CSV tuner resolves its config in this order:

1. `NCCL_TUNER_CONFIG_FILE` environment variable (if set)
2. `<librccl.so dir>/tuner/rccl_tuner_<arch>.csv` (adjacent to library, for development builds)
3. `<librccl.so dir>/tuner/rccl_tuner.csv` (adjacent to library, for development builds)
4. `<librccl.so dir>/../share/rccl/tuner/rccl_tuner_<arch>.csv` (relative share path, for installed RCCL)
5. `<librccl.so dir>/../share/rccl/tuner/rccl_tuner.csv` (relative share path, for installed RCCL)
6. `${ROCM_PATH}/share/rccl/tuner/rccl_tuner_<arch>.csv` (fallback, GPU-specific)
7. `${ROCM_PATH}/share/rccl/tuner/rccl_tuner.csv` (fallback, generic)
8. The embedded `rccl_tuner_<arch>.csv`, then the embedded `rccl_tuner.csv`; when the architecture is unknown, any embedded config

Steps 1-7 are overrides; step 8 is what ships in the binary.

At each disk location, if GPU architecture is unknown, the directory is scanned for any `rccl_tuner*.csv`
file. When the architecture *is* known, only an arch-specific or generic config is considered — another
architecture's file is never picked up. If neither exists, the tuner stays inactive.

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

To ignore only the configs compiled into the binary, while still honouring a CSV on disk:

```bash
export RCCL_TUNER_EMBEDDED_CONFIG=0
```

To disable the built-in CSV tuner entirely, embedded or not:

```bash
export NCCL_TUNER_PLUGIN=none
```
