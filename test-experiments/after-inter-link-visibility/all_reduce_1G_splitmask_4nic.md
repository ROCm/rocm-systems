# Test Evidence: After Inter-Link-Visibility (DSN Mapper + RCCL_BCM_LINKS_PATH)
# SPLIT_MASK=0x7, 4-NIC (8-GPU) — all_reduce & alltoall

**Date**: 2026-10-10
**Platform**: SMC300x MI300X (2 nodes)
**Nodes**: smc300x-ccs-aus-gpucc5a (ranks 0-7), smc300x-ccs-aus-gpuf273 (ranks 8-15)
**RCCL Version**: 2.32.3-develop (patched with RCCL_BCM_LINKS_PATH fallback)
**ROCm Version**: 7.2.4.0-93
**Network Plugin**: IB-CAST (AINIC RoCEv2)
**Inter-Link Visibility**: **ENABLED** via `rccl_dsn_mapper.sh --populate` + `RCCL_BCM_LINKS_PATH=/var/run/rccl_bcm_links`

---

## What Changed (vs Before)

1. **RCCL patch** (`src/os/linux.cc:ncclOsGetBcmLinks`): If default sysfs path
   `/sys/kernel/pci_switch_link/virtual_switch_links` doesn't exist, fall back to
   `RCCL_BCM_LINKS_PATH` env var directory.
2. **DSN mapper** (`rccl_dsn_mapper.sh --populate`): Discovers Broadcom PEX multi-host
   switch partitions via PCIe Device Serial Number, creates peer link entries at
   `/var/run/rccl_bcm_links/<busid>/<peer_busid>` on both nodes.
3. **Run command** adds `-x RCCL_BCM_LINKS_PATH=/var/run/rccl_bcm_links`.

---

## Test Configuration

```bash
NCCL_TESTS_SPLIT_MASK=0x7          # 8 independent 2-rank communicators
NCCL_IB_HCA=ionic_0,ionic_2,ionic_4,ionic_6   # 4-NIC filter
NCCL_MAX_NCHANNELS=64
NCCL_NCHANNELS_PER_NET_PEER=16
NCCL_P2P_NET_CHUNKSIZE=262144
RCCL_P2P_SHIFT_SIZE=0
NCCL_PXN_DISABLE=1
NCCL_GDR_FLUSH_DISABLE=1
RCCL_LL128_FORCE_ENABLE=1
RCCL_BCM_LINKS_PATH=/var/run/rccl_bcm_links   # NEW — DSN mapper output
```

```bash
# Run command
bash run-rccl.sh all_reduce 1G 1G 1 0
```

---

## Performance Result (1G quick check)

```
#       size         count      type   redop     time   algbw   busbw
  1073741824     536870912  bfloat16     sum   56981.6   18.84   18.84   (out-of-place)
                                                47492.9   22.61   22.61   (in-place)
# Avg bus bandwidth: 20.73 GB/s
```

---

## Topology Detection Summary

```
┌──────┬──────────┬────────────────────┬───────────────┬───────────┬──────────┐
│ Rank │ GPU BDF  │ NIC Assigned       │ Path Type     │ GDR       │ Channels │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 0    │ 05:00.0  │ ionic_6 (IB-CAST/0)│ PXB (same SW) │ ENABLED   │ 16       │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 1    │ 29:00.0  │ ionic_6 (IB-CAST/0)│ PXB (same SW) │ ENABLED   │ 16       │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 2    │ 49:00.0  │ ionic_4 (IB-CAST/1)│ PXB (same SW) │ ENABLED   │ 16       │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 3    │ 65:00.0  │ ionic_4 (IB-CAST/1)│ PXB (same SW) │ ENABLED   │ 16       │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 4    │ 85:00.0  │ ionic_2 (IB-CAST/2)│ PXB (same SW) │ ENABLED   │ 16       │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 5    │ a9:00.0  │ ionic_2 (IB-CAST/2)│ PXB (same SW) │ ENABLED   │ 16       │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 6    │ c9:00.0  │ ionic_0 (IB-CAST/3)│ PXB (same SW) │ ENABLED   │ 16       │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 7    │ e5:00.0  │ ionic_0 (IB-CAST/3)│ PXB (same SW) │ ENABLED   │ 16       │
└──────┴──────────┴────────────────────┴───────────────┴───────────┴──────────┘
```

**GDR ON**: 8 of 8 GPUs (100%) — all GPUs see same-physical-switch NIC via inter-link visibility
**GDR OFF**: 0 of 8 GPUs

---

## RCCL Log Evidence

### GDR Ring Status (per-rank)

```
Rank 0 [GPU 0]: Connected all rings, use ring PXN 0 GDR 1    <-- GDR ON
Rank 1 [GPU 1]: Connected all rings, use ring PXN 0 GDR 1    <-- GDR ON (was OFF)
Rank 2 [GPU 2]: Connected all rings, use ring PXN 0 GDR 1    <-- GDR ON (was OFF)
Rank 3 [GPU 3]: Connected all rings, use ring PXN 0 GDR 1    <-- GDR ON
Rank 4 [GPU 4]: Connected all rings, use ring PXN 0 GDR 1    <-- GDR ON
Rank 5 [GPU 5]: Connected all rings, use ring PXN 0 GDR 1    <-- GDR ON (was OFF)
Rank 6 [GPU 6]: Connected all rings, use ring PXN 0 GDR 1    <-- GDR ON (was OFF)
Rank 7 [GPU 7]: Connected all rings, use ring PXN 0 GDR 1    <-- GDR ON
```

All 8 ranks GDR ON (previously only ranks 0, 3, 4, 7 had GDR ON).

### All Ranks: GDRDMA Transport — Single NIC, Direct GPU-NIC DMA

Every rank now uses a single NIC with GDRDMA on all 16 channels (send + receive):

**Rank 0 (GPU 05:00.0) → ionic_6 (IB-CAST/0):**
```
Channel 00/0 : 1[0] -> 0[0] [receive] via NET/IB-CAST/0/GDRDMA/flush=Always
Channel 01/0 : 1[0] -> 0[0] [receive] via NET/IB-CAST/0/GDRDMA/flush=Always
  ... (all 16 channels use IB-CAST/0/GDRDMA)
Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB-CAST/0/GDRDMA
  ... (all 16 send channels use IB-CAST/0/GDRDMA)
```

**Rank 1 (GPU 29:00.0) → ionic_6 (IB-CAST/0):** ← WAS dual-NIC, NO GDRDMA
```
Channel 00/0 : 1[1] -> 0[1] [receive] via NET/IB-CAST/0/GDRDMA/flush=Always
  ... (all 16 channels use IB-CAST/0/GDRDMA — single NIC, same physical switch)
```

**Rank 2 (GPU 49:00.0) → ionic_4 (IB-CAST/1):** ← WAS dual-NIC, NO GDRDMA
```
Channel 00/0 : 1[2] -> 0[2] [receive] via NET/IB-CAST/1/GDRDMA/flush=Always
  ... (all 16 channels use IB-CAST/1/GDRDMA)
```

**Rank 3 (GPU 65:00.0) → ionic_4 (IB-CAST/1):**
```
Channel 00/0 : 1[3] -> 0[3] [receive] via NET/IB-CAST/1/GDRDMA/flush=Always
  ... (all 16 channels use IB-CAST/1/GDRDMA)
```

**Rank 4 (GPU 85:00.0) → ionic_2 (IB-CAST/2):**
```
Channel 00/0 : 1[4] -> 0[4] [receive] via NET/IB-CAST/2/GDRDMA/flush=Always
  ... (all 16 channels use IB-CAST/2/GDRDMA)
```

**Rank 5 (GPU a9:00.0) → ionic_2 (IB-CAST/2):** ← WAS dual-NIC, NO GDRDMA
```
Channel 00/0 : 1[5] -> 0[5] [receive] via NET/IB-CAST/2/GDRDMA/flush=Always
  ... (all 16 channels use IB-CAST/2/GDRDMA)
```

**Rank 6 (GPU c9:00.0) → ionic_0 (IB-CAST/3):** ← WAS dual-NIC, NO GDRDMA
```
Channel 00/0 : 1[6] -> 0[6] [receive] via NET/IB-CAST/3/GDRDMA/flush=Always
  ... (all 16 channels use IB-CAST/3/GDRDMA)
```

**Rank 7 (GPU e5:00.0) → ionic_0 (IB-CAST/3):**
```
Channel 00/0 : 1[7] -> 0[7] [receive] via NET/IB-CAST/3/GDRDMA/flush=Always
  ... (all 16 channels use IB-CAST/3/GDRDMA)
```

### Proxy Connection Count (uniform)

```
All 8 ranks: 33 proxy connections (connId 0-32)
```

Previously GDR-OFF ranks had 65 proxy connections — now uniform at 33.

---

## NIC Device Mapping (from RCCL logs)

```
Dev [0] = ionic_6  (PCI 0000:09:00.0, root port 0000:01:00.0, NUMA 0)
Dev [1] = ionic_4  (PCI 0000:69:00.0, root port 0000:61:00.0, NUMA 0)
Dev [2] = ionic_2  (PCI 0000:89:00.0, root port 0000:81:00.0, NUMA 1)
Dev [3] = ionic_0  (PCI 0000:e9:00.0, root port 0000:e1:00.0, NUMA 1)
```

## DSN Mapper Output (populated on both nodes)

```
/var/run/rccl_bcm_links/
├── 0000:01:00.0/
│   └── 0000:21:00.0
├── 0000:21:00.0/
│   └── 0000:01:00.0
├── 0000:41:00.0/
│   └── 0000:61:00.0
├── 0000:61:00.0/
│   └── 0000:41:00.0
├── 0000:81:00.0/
│   └── 0000:a1:00.0
├── 0000:a1:00.0/
│   └── 0000:81:00.0
├── 0000:c1:00.0/
│   └── 0000:e1:00.0
└── 0000:e1:00.0/
    └── 0000:c1:00.0
```

---

## GPU-to-NIC Affinity (Physical Switch Pairing)

```
┌──────────────────────────────────────────────────────────────────────┐
│  Physical Switch 1 (DSN: ...2e347e08)                               │
│    Partition A: root port 0000:01:00.0 → GPU0 (05:00.0), ionic_6   │
│    Partition B: root port 0000:21:00.0 → GPU1 (29:00.0), ionic_6   │
│    → Both GPUs share ionic_6 via same physical switch               │
├──────────────────────────────────────────────────────────────────────┤
│  Physical Switch 2 (DSN: ...9b33f108)                               │
│    Partition A: root port 0000:41:00.0 → GPU2 (49:00.0), ionic_4   │
│    Partition B: root port 0000:61:00.0 → GPU3 (65:00.0), ionic_4   │
│    → Both GPUs share ionic_4 via same physical switch               │
├──────────────────────────────────────────────────────────────────────┤
│  Physical Switch 3 (DSN: ...98b98808)                               │
│    Partition A: root port 0000:81:00.0 → GPU4 (85:00.0), ionic_2   │
│    Partition B: root port 0000:a1:00.0 → GPU5 (a9:00.0), ionic_2   │
│    → Both GPUs share ionic_2 via same physical switch               │
├──────────────────────────────────────────────────────────────────────┤
│  Physical Switch 4 (DSN: ...a00f9308)                               │
│    Partition A: root port 0000:c1:00.0 → GPU6 (c9:00.0), ionic_0   │
│    Partition B: root port 0000:e1:00.0 → GPU7 (e5:00.0), ionic_0   │
│    → Both GPUs share ionic_0 via same physical switch               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Observations (vs Before)

1. **100% GDR coverage** (was 50%): All 8 GPUs now have GDR enabled. Ranks 1, 2, 5, 6
   which previously had GDR OFF now correctly see their cross-partition NIC as PATH_PXB.

2. **Single NIC per GPU** (was dual-NIC for GDR-OFF ranks): Every rank uses exactly
   1 NIC with 16 channels. No more alternating dual-NIC assignment.

3. **Uniform 33 proxy connections** (was 65 for GDR-OFF ranks): All ranks have the
   same proxy connection count — no asymmetry.

4. **Correct physical switch affinity**: Each GPU pair shares the NIC on their common
   physical switch (GPU0+GPU1→ionic_6, GPU2+GPU3→ionic_4, GPU4+GPU5→ionic_2,
   GPU6+GPU7→ionic_0).

5. **GDRDMA on all channels**: Every send and receive channel across all 8 ranks uses
   GDRDMA. Receive channels use `flush=Always`.

---

## Performance Sweep: all_reduce (1K → 16G, 20 iters, 5 warmup)

```bash
bash run-rccl.sh all_reduce 1K 16G 20 5
```

```
#                                                              out-of-place                       in-place
#       size         count      type   redop     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
#        (B)    (elements)                       (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)
        1024           512  bfloat16     sum  3415.04    0.00    0.00       0  3243.96    0.00    0.00       0
        2048          1024  bfloat16     sum  2604.38    0.00    0.00       0  2681.58    0.00    0.00       0
        4096          2048  bfloat16     sum  2660.05    0.00    0.00       0  2428.55    0.00    0.00       0
        8192          4096  bfloat16     sum  2730.57    0.00    0.00       0  2194.88    0.00    0.00       0
       16384          8192  bfloat16     sum  3163.80    0.01    0.01       0  2753.19    0.01    0.01       0
       32768         16384  bfloat16     sum  2289.80    0.01    0.01       0  3061.89    0.01    0.01       0
       65536         32768  bfloat16     sum  2747.87    0.02    0.02       0  2483.23    0.03    0.03       0
      131072         65536  bfloat16     sum  3109.26    0.04    0.04       0  3288.65    0.04    0.04       0
      262144        131072  bfloat16     sum  3371.82    0.08    0.08       0  2460.06    0.11    0.11       0
      524288        262144  bfloat16     sum  3039.40    0.17    0.17       0  3236.85    0.16    0.16       0
     1048576        524288  bfloat16     sum  3573.09    0.29    0.29       0  2495.42    0.42    0.42       0
     2097152       1048576  bfloat16     sum  3233.48    0.65    0.65       0  3114.77    0.67    0.67       0
     4194304       2097152  bfloat16     sum  3516.06    1.19    1.19       0  2745.29    1.53    1.53       0
     8388608       4194304  bfloat16     sum  2465.36    3.40    3.40       0  2843.66    2.95    2.95       0
    16777216       8388608  bfloat16     sum  3051.81    5.50    5.50       0  3052.86    5.50    5.50       0
    33554432      16777216  bfloat16     sum  3153.14   10.64   10.64       0  3416.94    9.82    9.82       0
    67108864      33554432  bfloat16     sum  3675.42   18.26   18.26       0  3737.12   17.96   17.96       0
   134217728      67108864  bfloat16     sum  5744.87   23.36   23.36       0  5681.42   23.62   23.62       0
   268435456     134217728  bfloat16     sum  11321.0   23.71   23.71       0  11315.6   23.72   23.72       0
   536870912     268435456  bfloat16     sum  22319.2   24.05   24.05       0  22323.7   24.05   24.05       0
  1073741824     536870912  bfloat16     sum  44395.0   24.19   24.19       0  44469.9   24.15   24.15       0
  2147483648    1073741824  bfloat16     sum  88634.2   24.23   24.23       0  88508.1   24.26   24.26       0
  4294967296    2147483648  bfloat16     sum   176615   24.32   24.32       0   176754   24.30   24.30       0
  8589934592    4294967296  bfloat16     sum   352814   24.35   24.35       0   352613   24.36   24.36       0
 17179869184    8589934592  bfloat16     sum   704340   24.39   24.39       0   704821   24.37   24.37       0
```

**all_reduce saturated busBw** (256M+): ~24.0–24.4 GB/s (uniform in-place/out-of-place)

---

## Performance Sweep: alltoall (1K → 16G, 20 iters, 5 warmup)

```bash
bash run-rccl.sh alltoall 1K 16G 20 5
```

```
#                                                              out-of-place                       in-place
#       size         count      type   redop     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
#        (B)    (elements)                       (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)
        1024           256  bfloat16    none  3221.21    0.00    0.00       0  2124.92    0.00    0.00    N/A
        2048           512  bfloat16    none  3820.24    0.00    0.00       0  2455.58    0.00    0.00    N/A
        4096          1024  bfloat16    none  3166.91    0.00    0.00       0  2738.40    0.00    0.00    N/A
        8192          2048  bfloat16    none  3341.82    0.00    0.00       0  2851.95    0.00    0.00    N/A
       16384          4096  bfloat16    none  3216.69    0.01    0.00       0  2330.36    0.01    0.00    N/A
       32768          8192  bfloat16    none  3619.99    0.01    0.00       0  3150.94    0.01    0.01    N/A
       65536         16384  bfloat16    none  3209.86    0.02    0.01       0  2682.46    0.02    0.01    N/A
      131072         32768  bfloat16    none  3824.60    0.03    0.02       0  2415.84    0.05    0.03    N/A
      262144         65536  bfloat16    none  3114.55    0.08    0.04       0  2369.47    0.11    0.06    N/A
      524288        131072  bfloat16    none  3520.71    0.15    0.07       0  2200.79    0.24    0.12    N/A
     1048576        262144  bfloat16    none  3857.69    0.27    0.14       0  2471.13    0.42    0.21    N/A
     2097152        524288  bfloat16    none  3238.26    0.65    0.32       0  3218.01    0.65    0.33    N/A
     4194304       1048576  bfloat16    none  2997.94    1.40    0.70       0  2611.70    1.61    0.80    N/A
     8388608       2097152  bfloat16    none  4424.30    1.90    0.95       0  3408.37    2.46    1.23    N/A
    16777216       4194304  bfloat16    none  3197.97    5.25    2.62       0  3091.81    5.43    2.71    N/A
    33554432       8388608  bfloat16    none  3366.41    9.97    4.98       0  2624.72   12.78    6.39    N/A
    67108864      16777216  bfloat16    none  3049.83   22.00   11.00       0  3425.07   19.59    9.80    N/A
   134217728      33554432  bfloat16    none  4126.90   32.52   16.26       0  3205.85   41.87   20.93    N/A
   268435456      67108864  bfloat16    none  5946.09   45.14   22.57       0  6071.35   44.21   22.11    N/A
   536870912     134217728  bfloat16    none  11561.3   46.44   23.22       0  12006.8   44.71   22.36    N/A
  1073741824     268435456  bfloat16    none  22918.1   46.85   23.43       0  23807.9   45.10   22.55    N/A
  2147483648     536870912  bfloat16    none  45617.8   47.08   23.54       0  47407.7   45.30   22.65    N/A
  4294967296    1073741824  bfloat16    none  90923.2   47.24   23.62       0  94572.6   45.41   22.71    N/A
  8589934592    2147483648  bfloat16    none   181517   47.32   23.66       0   188724   45.52   22.76    N/A
 17179869184    4294967296  bfloat16    none   362988   47.33   23.66       0   377250   45.54   22.77    N/A
# Avg bus bandwidth    : 8.03
```

**alltoall saturated busBw** (256M+): ~23.2–23.7 GB/s (out-of-place), ~22.1–22.8 GB/s (in-place)

---

## Before vs After Comparison

```
┌─────────────┬──────────────────────────┬──────────────────────────┐
│ Metric      │ Before (no inter-link)   │ After (DSN mapper)       │
├─────────────┼──────────────────────────┼──────────────────────────┤
│ GDR ON      │ 4 of 8 GPUs (50%)        │ 8 of 8 GPUs (100%)      │
├─────────────┼──────────────────────────┼──────────────────────────┤
│ NICs/rank   │ 1 (GDR-ON), 2 (GDR-OFF) │ 1 (uniform)             │
├─────────────┼──────────────────────────┼──────────────────────────┤
│ Channels    │ 16 (GDR-ON), 32(GDR-OFF)│ 16 (uniform)            │
├─────────────┼──────────────────────────┼──────────────────────────┤
│ Proxy conns │ 33 (GDR-ON), 65(GDR-OFF)│ 33 (uniform)            │
├─────────────┼──────────────────────────┼──────────────────────────┤
│ Transport   │ GDRDMA / plain NET mix   │ GDRDMA (all channels)   │
├─────────────┼──────────────────────────┼──────────────────────────┤
│ AR busBw    │ ~24–26 GB/s (peak)       │ ~24.0–24.4 GB/s         │
│ (saturated) │ (asymmetric profile)     │ (uniform profile)       │
├─────────────┼──────────────────────────┼──────────────────────────┤
│ A2A busBw   │ ~23.1–23.6 GB/s          │ ~23.2–23.7 GB/s         │
│ (saturated) │                          │                         │
└─────────────┴──────────────────────────┴──────────────────────────┘
```

**Analysis**: Saturated bandwidth is comparable (~24 GB/s AR, ~23.5 GB/s A2A). The
before-fix "peak" of 26.86 GB/s at 2G was an artifact of GDR-OFF ranks using dual-rail
(2 NICs, 32 channels) which boosted throughput at certain message sizes despite CPU
bounce overhead. The after-fix numbers are more uniform and deterministic.

The primary win is architectural correctness: uniform GDR, uniform NIC assignment,
halved proxy connections, and elimination of the mixed-transport asymmetry.

---

## Log File References

```
Node 1 (smc300x-ccs-aus-gpucc5a):
  all_reduce 1G debug logs:
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.4030922.log  (GPU0)
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.4030923.log  (GPU1)
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.4030924.log  (GPU2)
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.4030925.log  (GPU3)
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.4030926.log  (GPU4)
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.4030927.log  (GPU5)
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.4030928.log  (GPU6)
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.4030929.log  (GPU7)
  all_reduce sweep logs:
    logs/rccl-all_reduce-1K-16G.smc300x-ccs-aus-gpucc5a.4038834-4038841.log
  alltoall sweep logs:
    logs/rccl-alltoall-1K-16G.smc300x-ccs-aus-gpucc5a.4049176-4049183.log
```

## RCCL Source Change

```
File: projects/rccl/src/os/linux.cc
Function: ncclOsGetBcmLinks()

- Default: /sys/kernel/pci_switch_link/virtual_switch_links (switch_discovery module)
- Fallback: RCCL_BCM_LINKS_PATH env var (only if default path doesn't exist)
```
