# LDS (Local Data Share) Layout for RCCL Device Linker

This document describes the **actual** layout of `ncclShmemData` in LDS (shared memory),
derived from disassembly analysis of the compiled `merged_device.elf`.

## Build Configuration

- `ENABLE_WARP_SPEED` - enabled
- `ENABLE_COLLTRACE` - enabled  
- `ENABLE_PROFILING` - enabled
- `ENABLE_FAULT_INJECTION` - enabled
- `NCCL_MAX_GROUPS` = 8 (512 threads / 64 warp size on AMD gfx942)
- `NCCL_MAX_ARITY` = 7 (NCCL_MAX_DIRECT_ARITY on AMD)
- `NCCL_MAX_CONNS` = 3

## Fixed LDS Size

From kernel metadata: **group_segment_fixed_size = 4944 bytes**

## Actual Layout from Assembly Analysis

### Verified Offsets from Disassembly

| Field | Actual Offset | How Verified |
|-------|---------------|--------------|
| `args.comm` (ptr) | 0 | ds_read_b64 v, v offset:0 |
| `channelId` | 96 | ds_read_b32 v, v offset:96 |
| `comm` | 48 | Structure calculation |
| **`channel`** | **192** | ds_read_b64 v, v offset:192 reads `channel.peers` |
| `channel.ring` | 200 | ds_read_b64 v, v offset:200 |
| **`warpComm`** | **608** | ds_write_b32 v, v offset:608 |
| **`warpChannel[0]`** | **624** | ds_write_b128 base + offset:624 |
| `warpChannel[1]` | 1008 | 624 + 384 stride |
| `warpChannelId[0]` | 2160 | ds_write_b32 v4, v0 offset:2160 |
| `groups[0].dsts` | 2216 | ds_read_b64 offset:2216 |
| `groups[1].dsts` | 2488 | ds_read_b64 offset:2488 |

### Structure Sizes

| Structure | Size (bytes) | Source |
|-----------|--------------|--------|
| ncclDevChannel | 384 (0x180) | Stride between warpChannel elements |
| ncclKernelComm | ~136 | Structure with COLLTRACE+PROFILING |

## ncclShmemData Layout (Actual)

| Offset | Field | Size | Notes |
|--------|-------|------|-------|
| 0 | `args` | 32 | ncclDevKernelArgs |
| 32 | `channelId` | 4 | |
| 36 | `aborted` | 4 | |
| 48 | `comm` | ~136 | ncclKernelComm (alignas 16) |
| **192** | **`channel`** | 384 | ncclDevChannel (alignas 16) |
| **192** | `channel.peers` | 8 | First field of channel |
| 576 | (end of channel) | - | |
| 576-607 | **UNKNOWN** | **32** | Unaccounted padding/data |
| **608** | **`warpComm`** | 4 | int |
| 612-623 | (padding) | 12 | alignas(16) for warpChannel |
| **624** | **`warpChannel[0]`** | 384 | ncclDevChannel |
| 1008 | `warpChannel[1]` | 384 | |
| 1392 | `warpChannel[2]` | 384 | |
| 1776 | `warpChannel[3]` | 384 | |
| 2160 | `warpChannel[4]` | 384 | |
| 2544 | `warpChannel[5]` | 384 | |
| 2928 | `warpChannel[6]` | 384 | |
| 3312 | `warpChannel[7]` | 384 | ends at 3696 |
| 3696 | `warpChannelId[8]` | 32 | int[8] |
| 3728 | remaining fields | ~1216 | groups, workStorage, etc. |

## ncclDevChannel Internal Layout (384 bytes)

| Offset | Field | Size | Notes |
|--------|-------|------|-------|
| 0 | `peers` | 8 | ncclDevChannelPeer** - **KEY POINTER** |
| 8 | `ring` | 24 | ncclRing |
| 32 | `tree` | 20 | ncclTree |
| 52 | `collnetChain` | 20 | ncclTree |
| 72 | `collnetDirect` | 108 | ncclDirect |
| 180 | `binTree` | 20 | ncclTree |
| 200 | `nvls` | 160 | ncclNvls |
| 360 | `workFifoDone` | 8 | uint32_t* |
| 368 | `workCounter` | 8 | uint64_t |

## Critical Debugging Offsets

| What to Check | LDS Offset | Notes |
|---------------|------------|-------|
| `channel.peers` | **192** | Main channel peer pointer |
| `warpChannel[0].peers` | **624** | Warp 0 peer pointer |
| `warpChannel[1].peers` | 1008 | Warp 1 peer pointer |
| `warpChannel[N].peers` | 624 + N*384 | General formula |

## Assembly Access Pattern for warpChannel

The specialized kernels compute warpChannel address as:
```asm
v_lshrrev_b32 v7, 6, v22           ; v7 = threadIdx >> 6 = warpId
v_lshlrev_b32 v4, 2, v10           ; v4 = warpId * 4
v_mul_u32_u24 v5, 0x17c, v10       ; v5 = warpId * 380
v_add3_u32 v4, v4, v5, v8          ; v4 = warpId*384 + laneId*16
ds_read_b64 v[16:17], v7 offset:624 ; read from LDS[warpId*384 + 624]
```

## Known Issue: 32-byte Gap

There's an **unexplained 32-byte gap** between:
- `channel` end (192 + 384 = 576)
- `warpComm` start (608)

Possible causes:
1. Compiler-inserted alignment padding
2. Hidden conditional field not visible in headers
3. Build configuration difference

**This gap is consistent** - both dispatcher and specialized kernels use the same
offsets (608 for warpComm, 624 for warpChannel[0]).

## Dispatcher → Specialized Kernel Data Flow

1. **Dispatcher loads** `ncclKernelCommAndChannels` from global memory
2. **Dispatcher writes** per-warp channel to LDS:
   - Destination: `warpChannel[localWarpId]` at offset `624 + localWarpId * 384`
   - Source: `channels[channelId]` from global `ncclKernelCommAndChannels`
3. **Specialized kernel reads**:
   - `warpChannel[localWarpId].peers` from LDS offset `624 + warpId*384`
4. **Peers pointer** → global `ncclDevChannelPeer` array → `ncclConnInfo` buffers
