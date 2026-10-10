# Test Evidence: Before Inter-Link-Visibility
# SPLIT_MASK=0x7, 4-NIC (8-GPU) — all_reduce & alltoall

**Date**: 2026-10-10
**Platform**: SMC300x MI300X (2 nodes)
**Nodes**: smc300x-ccs-aus-gpucc5a (ranks 0-7), smc300x-ccs-aus-gpuf273 (ranks 8-15)
**RCCL Version**: 2.32.3-develop:aa01e4e
**ROCm Version**: 7.2.4.0-93
**Network Plugin**: IB-CAST (AINIC RoCEv2)
**Inter-Link Visibility**: **NOT AVAILABLE** (`switch_discovery` module not loaded, no DSN mapper)

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
```

```bash
# Run command
bash run-rccl.sh all_reduce 1G 1G 1 0
```

---

## Performance Result

```
#       size         count      type   redop     time   algbw   busbw
  1073741824     536870912  bfloat16     sum   42091.8   25.51   25.51   (in-place)
                                                51895.5   20.69   20.69   (out-of-place)
# Avg bus bandwidth: 23.1 GB/s
```

---

## Topology Detection Summary

```
┌──────┬──────────┬────────────────────┬───────────────┬───────────┬──────────┐
│ Rank │ GPU BDF  │ NIC(s) Assigned    │ Path Type     │ GDR       │ Channels │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 0    │ 05:00.0  │ ionic_6            │ PXB (same SW) │ ENABLED   │ 16       │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 1    │ 29:00.0  │ ionic_6 + ionic_4  │ PHB (via CPU) │ DISABLED  │ 32       │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 2    │ 49:00.0  │ ionic_6 + ionic_4  │ PHB (via CPU) │ DISABLED  │ 32       │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 3    │ 65:00.0  │ ionic_4            │ PXB (same SW) │ ENABLED   │ 16       │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 4    │ 85:00.0  │ ionic_2            │ PXB (same SW) │ ENABLED   │ 16       │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 5    │ a9:00.0  │ ionic_2 + ionic_0  │ PHB (via CPU) │ DISABLED  │ 32       │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 6    │ c9:00.0  │ ionic_2 + ionic_0  │ PHB (via CPU) │ DISABLED  │ 32       │
├──────┼──────────┼────────────────────┼───────────────┼───────────┼──────────┤
│ 7    │ e5:00.0  │ ionic_0            │ PXB (same SW) │ ENABLED   │ 16       │
└──────┴──────────┴────────────────────┴───────────────┴───────────┴──────────┘
```

**GDR ON**: 4 of 8 GPUs (ranks 0, 3, 4, 7) — same-partition NIC available
**GDR OFF**: 4 of 8 GPUs (ranks 1, 2, 5, 6) — no same-partition NIC, PHB path to nearest NICs

---

## RCCL Log Evidence

### GDR Ring Status (per-rank)

```
Rank 0 [GPU 0]: Connected all rings, use ring PXN 0 GDR 1    <-- GDR ON
Rank 1 [GPU 1]: Connected all rings, use ring PXN 0 GDR 0    <-- GDR OFF
Rank 2 [GPU 2]: Connected all rings, use ring PXN 0 GDR 0    <-- GDR OFF
Rank 3 [GPU 3]: Connected all rings, use ring PXN 0 GDR 1    <-- GDR ON
Rank 4 [GPU 4]: Connected all rings, use ring PXN 0 GDR 1    <-- GDR ON
Rank 5 [GPU 5]: Connected all rings, use ring PXN 0 GDR 0    <-- GDR OFF
Rank 6 [GPU 6]: Connected all rings, use ring PXN 0 GDR 0    <-- GDR OFF
Rank 7 [GPU 7]: Connected all rings, use ring PXN 0 GDR 1    <-- GDR ON
```

All ranks report `cuMemGdrSupport 1` — hardware supports GDR, but RCCL disables it for PHB paths.

### GDR-ON Ranks: GDRDMA Transport (Ranks 0, 3, 4, 7)

Single NIC, direct GPU-NIC DMA:

**Rank 0 → ionic_6 (IB-CAST/0):**
```
Channel 00/0 : 1[0] -> 0[0] [receive] via NET/IB-CAST/0/GDRDMA/flush=Always
Channel 01/0 : 1[0] -> 0[0] [receive] via NET/IB-CAST/0/GDRDMA/flush=Always
  ... (all 16 channels use IB-CAST/0/GDRDMA)
Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB-CAST/0/GDRDMA
  ... (all 16 send channels use IB-CAST/0/GDRDMA)
```

**Rank 3 → ionic_4 (IB-CAST/1):**
```
Channel 00/0 : 1[3] -> 0[3] [receive] via NET/IB-CAST/1/GDRDMA/flush=Always
  ... (all 16 channels use IB-CAST/1/GDRDMA)
```

**Rank 4 → ionic_2 (IB-CAST/2):**
```
Channel 00/0 : 1[4] -> 0[4] [receive] via NET/IB-CAST/2/GDRDMA/flush=Always
  ... (all 16 channels use IB-CAST/2/GDRDMA)
```

**Rank 7 → ionic_0 (IB-CAST/3):**
```
Channel 00/0 : 1[7] -> 0[7] [receive] via NET/IB-CAST/3/GDRDMA/flush=Always
  ... (all 16 channels use IB-CAST/3/GDRDMA)
```

### GDR-OFF Ranks: Non-GDRDMA Transport (Ranks 1, 2, 5, 6)

Dual-NIC, CPU bounce buffer, alternating NIC assignment:

**Rank 1 (GPU1) — alternates ionic_4 (IB-CAST/1) and ionic_6 (IB-CAST/0):**
```
Channel 00/0 : 1[1] -> 0[1] [receive] via NET/IB-CAST/1
Channel 01/0 : 1[1] -> 0[1] [receive] via NET/IB-CAST/0
Channel 02/0 : 1[1] -> 0[1] [receive] via NET/IB-CAST/1
Channel 03/0 : 1[1] -> 0[1] [receive] via NET/IB-CAST/0
  ... (alternating pattern across all 16 recv channels, NO GDRDMA suffix)
```

**Rank 2 (GPU2) — alternates ionic_6 (IB-CAST/0) and ionic_4 (IB-CAST/1):**
```
Channel 00/0 : 1[2] -> 0[2] [receive] via NET/IB-CAST/0
Channel 01/0 : 1[2] -> 0[2] [receive] via NET/IB-CAST/1
  ... (alternating pattern, NO GDRDMA)
```

**Rank 5 (GPU5) — alternates ionic_0 (IB-CAST/3) and ionic_2 (IB-CAST/2):**
```
Channel 00/0 : 1[5] -> 0[5] [receive] via NET/IB-CAST/3
Channel 01/0 : 1[5] -> 0[5] [receive] via NET/IB-CAST/2
  ... (alternating pattern, NO GDRDMA)
```

**Rank 6 (GPU6) — alternates ionic_2 (IB-CAST/2) and ionic_0 (IB-CAST/3):**
```
Channel 00/0 : 1[6] -> 0[6] [receive] via NET/IB-CAST/2
Channel 01/0 : 1[6] -> 0[6] [receive] via NET/IB-CAST/3
  ... (alternating pattern, NO GDRDMA)
```

### Proxy Connection Count Difference

```
GDR-ON  ranks (0, 3, 4, 7): 33 proxy connections (connId 0-32)
GDR-OFF ranks (1, 2, 5, 6): 65 proxy connections (connId 0-64)
```

GDR-OFF ranks require ~2x proxy connections because both send and receive directions
need proxy-buffered host memory copies without GDRDMA.

### GDR Flush (runtime evidence)

Only GDR-ON ranks emit this line during data transfer:
```
Rank 0: recvProxyProgress: issued GDR flush
Rank 3: recvProxyProgress: issued GDR flush
Rank 4: recvProxyProgress: issued GDR flush
Rank 7: recvProxyProgress: issued GDR flush
```
Ranks 1, 2, 5, 6 have NO "GDR flush" log — consistent with GDR being off.

---

## NIC Device Mapping (from RCCL logs)

```
Dev [0] = ionic_6  (PCI 0000:09:00.0, root port 0000:01:00.0, NUMA 0)
Dev [1] = ionic_4  (PCI 0000:69:00.0, root port 0000:61:00.0, NUMA 0)
Dev [2] = ionic_2  (PCI 0000:89:00.0, root port 0000:81:00.0, NUMA 1)
Dev [3] = ionic_0  (PCI 0000:e9:00.0, root port 0000:e1:00.0, NUMA 1)
```

---

## Key Observations

1. **50% GDR loss**: Only 4 of 8 GPUs have GDR enabled — the 4 GPUs whose same-partition
   NIC survived the 4-NIC mask (GPU0↔ionic_6, GPU3↔ionic_4, GPU4↔ionic_2, GPU7↔ionic_0).

2. **PHB path for orphaned GPUs**: GPU1, GPU2, GPU5, GPU6 lost their paired NICs
   (ionic_7, ionic_5, ionic_3, ionic_1). RCCL classifies all remaining same-NUMA NICs
   as equidistant at PATH_PHB — it cannot distinguish cross-partition (same physical
   switch, internal fabric) from cross-switch (different physical switch, through CPU).

3. **Dual-rail compensation**: GDR-OFF ranks use 2 NICs (alternating) with 32 channels
   vs 1 NIC with 16 channels for GDR-ON ranks. This attempts to recover bandwidth
   but adds CPU bounce buffer overhead.

4. **Expected fix**: With inter-link visibility (via switch_discovery or DSN mapper),
   cross-partition paths would be classified as PATH_PXB instead of PATH_PHB, enabling
   GDR for all 8 GPUs and assigning each GPU to its same-physical-switch NIC.

---

---

## Performance Sweep: all_reduce (1K → 16G, 20 iters, 5 warmup)

```bash
bash run-rccl.sh all_reduce 1K 16G 20 5
```

```
#                                                              out-of-place                       in-place
#       size         count      type   redop     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
#        (B)    (elements)                       (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)
        1024           512  bfloat16     sum  3132.98    0.00    0.00       0  2576.29    0.00    0.00       0
        2048          1024  bfloat16     sum  2912.06    0.00    0.00       0  3440.55    0.00    0.00       0
        4096          2048  bfloat16     sum  3478.72    0.00    0.00       0  2574.21    0.00    0.00       0
        8192          4096  bfloat16     sum  2676.78    0.00    0.00       0  2724.02    0.00    0.00       0
       16384          8192  bfloat16     sum  3435.58    0.00    0.00       0  4144.87    0.00    0.00       0
       32768         16384  bfloat16     sum  3799.47    0.01    0.01       0  2820.63    0.01    0.01       0
       65536         32768  bfloat16     sum  3352.35    0.02    0.02       0  2852.58    0.02    0.02       0
      131072         65536  bfloat16     sum  2763.51    0.05    0.05       0  2518.01    0.05    0.05       0
      262144        131072  bfloat16     sum  3100.13    0.08    0.08       0  3187.03    0.08    0.08       0
      524288        262144  bfloat16     sum  2494.95    0.21    0.21       0  2551.92    0.21    0.21       0
     1048576        524288  bfloat16     sum  3037.66    0.35    0.35       0  3290.00    0.32    0.32       0
     2097152       1048576  bfloat16     sum  3388.77    0.62    0.62       0  3299.57    0.64    0.64       0
     4194304       2097152  bfloat16     sum  3422.88    1.23    1.23       0  3609.78    1.16    1.16       0
     8388608       4194304  bfloat16     sum  3458.73    2.43    2.43       0  3007.23    2.79    2.79       0
    16777216       8388608  bfloat16     sum  2810.43    5.97    5.97       0  2847.16    5.89    5.89       0
    33554432      16777216  bfloat16     sum  3209.23   10.46   10.46       0  3566.82    9.41    9.41       0
    67108864      33554432  bfloat16     sum  4638.60   14.47   14.47       0  4460.21   15.05   15.05       0
   134217728      67108864  bfloat16     sum  6354.51   21.12   21.12       0  6337.03   21.18   21.18       0
   268435456     134217728  bfloat16     sum  12200.0   22.00   22.00       0  11948.4   22.47   22.47       0
   536870912     268435456  bfloat16     sum  23789.3   22.57   22.57       0  23471.5   22.87   22.87       0
  1073741824     536870912  bfloat16     sum  46484.4   23.10   23.10       0  44715.7   24.01   24.01       0
  2147483648    1073741824  bfloat16     sum  80483.3   26.68   26.68       0  79948.6   26.86   26.86       0
  4294967296    2147483648  bfloat16     sum   173541   24.75   24.75       0   175849   24.42   24.42       0
  8589934592    4294967296  bfloat16     sum   349937   24.55   24.55       0   345461   24.87   24.87       0
 17179869184    8589934592  bfloat16     sum   694430   24.74   24.74       0   688131   24.96   24.96       0
```

**all_reduce peak busBw**: ~26.86 GB/s (in-place, 2G), ~24.96 GB/s (in-place, 16G)

---

## Performance Sweep: alltoall (1K → 16G, 20 iters, 5 warmup)

```bash
bash run-rccl.sh alltoall 1K 16G 20 5
```

```
#                                                              out-of-place                       in-place
#       size         count      type   redop     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
#        (B)    (elements)                       (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)
        1024           256  bfloat16    none  4009.86    0.00    0.00       0  2892.34    0.00    0.00    N/A
        2048           512  bfloat16    none  4068.69    0.00    0.00       0  3792.33    0.00    0.00    N/A
        4096          1024  bfloat16    none  4533.09    0.00    0.00       0  2987.36    0.00    0.00    N/A
        8192          2048  bfloat16    none  3828.75    0.00    0.00       0  2524.43    0.00    0.00    N/A
       16384          4096  bfloat16    none  3793.45    0.00    0.00       0  3731.08    0.00    0.00    N/A
       32768          8192  bfloat16    none  4691.37    0.01    0.00       0  2949.39    0.01    0.01    N/A
       65536         16384  bfloat16    none  3897.90    0.02    0.01       0  2818.82    0.02    0.01    N/A
      131072         32768  bfloat16    none  3550.35    0.04    0.02       0  2987.64    0.04    0.02    N/A
      262144         65536  bfloat16    none  3891.39    0.07    0.03       0  3000.36    0.09    0.04    N/A
      524288        131072  bfloat16    none  4168.84    0.13    0.06       0  2851.91    0.18    0.09    N/A
     1048576        262144  bfloat16    none  4009.41    0.26    0.13       0  2912.24    0.36    0.18    N/A
     2097152        524288  bfloat16    none  3191.58    0.66    0.33       0  2437.58    0.86    0.43    N/A
     4194304       1048576  bfloat16    none  4241.83    0.99    0.49       0  2478.21    1.69    0.85    N/A
     8388608       2097152  bfloat16    none  4188.21    2.00    1.00       0  2893.92    2.90    1.45    N/A
    16777216       4194304  bfloat16    none  3411.37    4.92    2.46       0  2932.64    5.72    2.86    N/A
    33554432       8388608  bfloat16    none  6329.53    5.30    2.65       0  2946.73   11.39    5.69    N/A
    67108864      16777216  bfloat16    none  13539.7    4.96    2.48       0  13670.1    4.91    2.45    N/A
   134217728      33554432  bfloat16    none  26742.1    5.02    2.51       0  26819.1    5.00    2.50    N/A
   268435456      67108864  bfloat16    none  32576.4    8.24    4.12       0  20824.4   12.89    6.45    N/A
   536870912     134217728  bfloat16    none  20714.7   25.92   12.96       0  76433.9    7.02    3.51    N/A
  1073741824     268435456  bfloat16    none  86446.4   12.42    6.21       0  74076.9   14.49    7.25    N/A
  2147483648     536870912  bfloat16    none   248862    8.63    4.31       0   225335    9.53    4.77    N/A
  4294967296    1073741824  bfloat16    none   207120   20.74   10.37       0   403977   10.63    5.32    N/A
  8589934592    2147483648  bfloat16    none   884954    9.71    4.85       0   205553   41.79   20.89    N/A
 17179869184    4294967296  bfloat16    none   371903   46.19   23.10       0   376910   45.58   22.79    N/A
# Avg bus bandwidth: 3.31 GB/s
```

**alltoall peak busBw**: ~23.10 GB/s (out-of-place, 16G), ~22.79 GB/s (in-place, 16G)

---

## Log File References

```
Node 1 (smc300x-ccs-aus-gpucc5a):
  all_reduce 1G debug logs:
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.3706296.log  (Rank 0, GPU0)
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.3706297.log  (Rank 1, GPU1)
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.3706298.log  (Rank 2, GPU2)
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.3706299.log  (Rank 3, GPU3)
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.3706300.log  (Rank 4, GPU4)
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.3706301.log  (Rank 5, GPU5)
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.3706302.log  (Rank 6, GPU6)
    logs/rccl-all_reduce-1G-1G.smc300x-ccs-aus-gpucc5a.3706303.log  (Rank 7, GPU7)
  all_reduce sweep logs:
    logs/rccl-all_reduce-1K-16G.smc300x-ccs-aus-gpucc5a.3809129-3809136.log
  alltoall sweep logs:
    logs/rccl-alltoall-1K-16G.smc300x-ccs-aus-gpucc5a.3811111-3811118.log
```
