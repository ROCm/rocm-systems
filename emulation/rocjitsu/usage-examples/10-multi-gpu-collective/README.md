# Example 10: Multi-GPU RCCL Collectives

## Objective

Debug multi-GPU applications using RCCL for collective operations.

## Prerequisites

- RCCL library
- 2-GPU configuration

## Files

- `src/allreduce.cpp` - AllReduce collective debugging
- `Makefile`

## Quick Start

```bash
make
make run-2gpu  # Use 2-GPU configuration
```

## Configuration

Uses `amdgpu_cdna4_kmd_2gpu.json` for dual GPU setup.

## Common Collectives

- **AllReduce** - Reduce and broadcast
- **Broadcast** - One-to-all
- **AllGather** - Gather from all
- **ReduceScatter** - Reduce and scatter

## Debugging Tips

- Ensure all ranks participate
- Check communicator initialization
- Verify buffer sizes match
- Use logging to trace execution