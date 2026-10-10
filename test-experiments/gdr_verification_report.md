# GDR ON/OFF Verification Report — All 10 Collectives

**Platform**: SMC300x MI300X (2 nodes, 8 GPUs/node, 4 NICs/node)
**Config**: `NCCL_TESTS_SPLIT_MASK=0x7`, `NCCL_MAX_NCHANNELS=16`, `NCCL_DEBUG=info`
**Message Size**: 1G (single size, 1 iteration, 0 warmup)
**Before-fix**: 4 NICs, no inter-switch-link visibility
**After-fix**: 4 NICs + DSN mapper inter-switch-links (`RCCL_BCM_LINKS_PATH`)
**Date**: 2026-10-10

## Per-GPU GDR & Transport Detail (Node 1: `gpucc5a`)

```
┌─────────────────┬─────┬──────────┬──────────────────────────────┬──────────────────────────────┐
│ Collective      │ GPU │ GPU BDF  │ Before-Fix                   │ After-Fix                    │
│                 │     │          │ GDR  NICs  Transport          │ GDR  NICs  Transport          │
├─────────────────┼─────┼──────────┼──────────────────────────────┼──────────────────────────────┤
│ all_reduce      │  0  │  05:00.0 │ ON   1[0  ] GDRDMA (32ch)    │ ON   1[0  ] GDRDMA (32ch)    │
│                 │  1  │  29:00.0 │ OFF  2[0,1] NET    (32ch)    │ ON   1[0  ] GDRDMA (32ch)    │
│                 │  2  │  49:00.0 │ OFF  2[0,1] NET    (32ch)    │ ON   1[1  ] GDRDMA (32ch)    │
│                 │  3  │  65:00.0 │ ON   1[1  ] GDRDMA (32ch)    │ ON   1[1  ] GDRDMA (32ch)    │
│                 │  4  │  85:00.0 │ ON   1[2  ] GDRDMA (32ch)    │ ON   1[2  ] GDRDMA (32ch)    │
│                 │  5  │  a9:00.0 │ OFF  2[2,3] NET    (32ch)    │ ON   1[2  ] GDRDMA (32ch)    │
│                 │  6  │  c9:00.0 │ OFF  2[2,3] NET    (32ch)    │ ON   1[3  ] GDRDMA (32ch)    │
│                 │  7  │  e5:00.0 │ ON   1[3  ] GDRDMA (32ch)    │ ON   1[3  ] GDRDMA (32ch)    │
├─────────────────┼─────┼──────────┼──────────────────────────────┼──────────────────────────────┤
│ alltoall        │  0  │  05:00.0 │ ON   1[0  ] GDRDMA (64ch)    │ ON   1[0  ] GDRDMA (64ch)    │
│                 │  1  │  29:00.0 │ OFF  2[0,1] NET    (64ch)    │ ON   1[0  ] GDRDMA (64ch)    │
│                 │  2  │  49:00.0 │ OFF  2[0,1] NET    (64ch)    │ ON   1[1  ] GDRDMA (64ch)    │
│                 │  3  │  65:00.0 │ ON   1[1  ] GDRDMA (64ch)    │ ON   1[1  ] GDRDMA (64ch)    │
│                 │  4  │  85:00.0 │ ON   1[2  ] GDRDMA (64ch)    │ ON   1[2  ] GDRDMA (64ch)    │
│                 │  5  │  a9:00.0 │ OFF  2[2,3] NET    (64ch)    │ ON   1[2  ] GDRDMA (64ch)    │
│                 │  6  │  c9:00.0 │ OFF  2[2,3] NET    (64ch)    │ ON   1[3  ] GDRDMA (64ch)    │
│                 │  7  │  e5:00.0 │ ON   1[3  ] GDRDMA (64ch)    │ ON   1[3  ] GDRDMA (64ch)    │
├─────────────────┼─────┼──────────┼──────────────────────────────┼──────────────────────────────┤
│ alltoallv       │  0  │  05:00.0 │ ON   1[0  ] GDRDMA (64ch)    │ ON   1[0  ] GDRDMA (64ch)    │
│                 │  1  │  29:00.0 │ OFF  2[0,1] NET    (64ch)    │ ON   1[0  ] GDRDMA (64ch)    │
│                 │  2  │  49:00.0 │ OFF  2[0,1] NET    (64ch)    │ ON   1[1  ] GDRDMA (64ch)    │
│                 │  3  │  65:00.0 │ ON   1[1  ] GDRDMA (64ch)    │ ON   1[1  ] GDRDMA (64ch)    │
│                 │  4  │  85:00.0 │ ON   1[2  ] GDRDMA (64ch)    │ ON   1[2  ] GDRDMA (64ch)    │
│                 │  5  │  a9:00.0 │ OFF  2[2,3] NET    (64ch)    │ ON   1[2  ] GDRDMA (64ch)    │
│                 │  6  │  c9:00.0 │ OFF  2[2,3] NET    (64ch)    │ ON   1[3  ] GDRDMA (64ch)    │
│                 │  7  │  e5:00.0 │ ON   1[3  ] GDRDMA (64ch)    │ ON   1[3  ] GDRDMA (64ch)    │
├─────────────────┼─────┼──────────┼──────────────────────────────┼──────────────────────────────┤
│ broadcast       │  0  │  05:00.0 │ ON   1[0  ] GDRDMA (32ch)    │ ON   1[0  ] GDRDMA (32ch)    │
│                 │  1  │  29:00.0 │ OFF  2[0,1] NET    (32ch)    │ ON   1[0  ] GDRDMA (32ch)    │
│                 │  2  │  49:00.0 │ OFF  2[0,1] NET    (32ch)    │ ON   1[1  ] GDRDMA (32ch)    │
│                 │  3  │  65:00.0 │ ON   1[1  ] GDRDMA (32ch)    │ ON   1[1  ] GDRDMA (32ch)    │
│                 │  4  │  85:00.0 │ ON   1[2  ] GDRDMA (32ch)    │ ON   1[2  ] GDRDMA (32ch)    │
│                 │  5  │  a9:00.0 │ OFF  2[2,3] NET    (32ch)    │ ON   1[2  ] GDRDMA (32ch)    │
│                 │  6  │  c9:00.0 │ OFF  2[2,3] NET    (32ch)    │ ON   1[3  ] GDRDMA (32ch)    │
│                 │  7  │  e5:00.0 │ ON   1[3  ] GDRDMA (32ch)    │ ON   1[3  ] GDRDMA (32ch)    │
├─────────────────┼─────┼──────────┼──────────────────────────────┼──────────────────────────────┤
│ reduce_scatter  │  0  │  05:00.0 │ ON   1[0  ] GDRDMA (32ch)    │ ON   1[0  ] GDRDMA (32ch)    │
│                 │  1  │  29:00.0 │ OFF  2[0,1] NET    (32ch)    │ ON   1[0  ] GDRDMA (32ch)    │
│                 │  2  │  49:00.0 │ OFF  2[0,1] NET    (32ch)    │ ON   1[1  ] GDRDMA (32ch)    │
│                 │  3  │  65:00.0 │ ON   1[1  ] GDRDMA (32ch)    │ ON   1[1  ] GDRDMA (32ch)    │
│                 │  4  │  85:00.0 │ ON   1[2  ] GDRDMA (32ch)    │ ON   1[2  ] GDRDMA (32ch)    │
│                 │  5  │  a9:00.0 │ OFF  2[2,3] NET    (32ch)    │ ON   1[2  ] GDRDMA (32ch)    │
│                 │  6  │  c9:00.0 │ OFF  2[2,3] NET    (32ch)    │ ON   1[3  ] GDRDMA (32ch)    │
│                 │  7  │  e5:00.0 │ ON   1[3  ] GDRDMA (32ch)    │ ON   1[3  ] GDRDMA (32ch)    │
├─────────────────┼─────┼──────────┼──────────────────────────────┼──────────────────────────────┤
│ all_gather      │  0  │  05:00.0 │ ON   1[0  ] GDRDMA (32ch)    │ ON   1[0  ] GDRDMA (32ch)    │
│                 │  1  │  29:00.0 │ OFF  2[0,1] NET    (32ch)    │ ON   1[0  ] GDRDMA (32ch)    │
│                 │  2  │  49:00.0 │ OFF  2[0,1] NET    (32ch)    │ ON   1[1  ] GDRDMA (32ch)    │
│                 │  3  │  65:00.0 │ ON   1[1  ] GDRDMA (32ch)    │ ON   1[1  ] GDRDMA (32ch)    │
│                 │  4  │  85:00.0 │ ON   1[2  ] GDRDMA (32ch)    │ ON   1[2  ] GDRDMA (32ch)    │
│                 │  5  │  a9:00.0 │ OFF  2[2,3] NET    (32ch)    │ ON   1[2  ] GDRDMA (32ch)    │
│                 │  6  │  c9:00.0 │ OFF  2[2,3] NET    (32ch)    │ ON   1[3  ] GDRDMA (32ch)    │
│                 │  7  │  e5:00.0 │ ON   1[3  ] GDRDMA (32ch)    │ ON   1[3  ] GDRDMA (32ch)    │
├─────────────────┼─────┼──────────┼──────────────────────────────┼──────────────────────────────┤
│ reduce          │  0  │  05:00.0 │ ON   1[0  ] GDRDMA (32ch)    │ ON   1[0  ] GDRDMA (32ch)    │
│                 │  1  │  29:00.0 │ OFF  2[0,1] NET    (32ch)    │ ON   1[0  ] GDRDMA (32ch)    │
│                 │  2  │  49:00.0 │ OFF  2[0,1] NET    (32ch)    │ ON   1[1  ] GDRDMA (32ch)    │
│                 │  3  │  65:00.0 │ ON   1[1  ] GDRDMA (32ch)    │ ON   1[1  ] GDRDMA (32ch)    │
│                 │  4  │  85:00.0 │ ON   1[2  ] GDRDMA (32ch)    │ ON   1[2  ] GDRDMA (32ch)    │
│                 │  5  │  a9:00.0 │ OFF  2[2,3] NET    (32ch)    │ ON   1[2  ] GDRDMA (32ch)    │
│                 │  6  │  c9:00.0 │ OFF  2[2,3] NET    (32ch)    │ ON   1[3  ] GDRDMA (32ch)    │
│                 │  7  │  e5:00.0 │ ON   1[3  ] GDRDMA (32ch)    │ ON   1[3  ] GDRDMA (32ch)    │
├─────────────────┼─────┼──────────┼──────────────────────────────┼──────────────────────────────┤
│ scatter         │  0  │  05:00.0 │ ON   1[0  ] GDRDMA (48ch)    │ ON   1[0  ] GDRDMA (48ch)    │
│                 │  1  │  29:00.0 │ OFF  2[0,1] NET    (48ch)    │ ON   1[0  ] GDRDMA (48ch)    │
│                 │  2  │  49:00.0 │ OFF  2[0,1] NET    (48ch)    │ ON   1[1  ] GDRDMA (48ch)    │
│                 │  3  │  65:00.0 │ ON   1[1  ] GDRDMA (48ch)    │ ON   1[1  ] GDRDMA (48ch)    │
│                 │  4  │  85:00.0 │ ON   1[2  ] GDRDMA (48ch)    │ ON   1[2  ] GDRDMA (48ch)    │
│                 │  5  │  a9:00.0 │ OFF  2[2,3] NET    (48ch)    │ ON   1[2  ] GDRDMA (48ch)    │
│                 │  6  │  c9:00.0 │ OFF  2[2,3] NET    (48ch)    │ ON   1[3  ] GDRDMA (48ch)    │
│                 │  7  │  e5:00.0 │ ON   1[3  ] GDRDMA (48ch)    │ ON   1[3  ] GDRDMA (48ch)    │
├─────────────────┼─────┼──────────┼──────────────────────────────┼──────────────────────────────┤
│ gather          │  0  │  05:00.0 │ ON   1[0  ] GDRDMA (48ch)    │ ON   1[0  ] GDRDMA (48ch)    │
│                 │  1  │  29:00.0 │ OFF  2[0,1] NET    (48ch)    │ ON   1[0  ] GDRDMA (48ch)    │
│                 │  2  │  49:00.0 │ OFF  2[0,1] NET    (48ch)    │ ON   1[1  ] GDRDMA (48ch)    │
│                 │  3  │  65:00.0 │ ON   1[1  ] GDRDMA (48ch)    │ ON   1[1  ] GDRDMA (48ch)    │
│                 │  4  │  85:00.0 │ ON   1[2  ] GDRDMA (48ch)    │ ON   1[2  ] GDRDMA (48ch)    │
│                 │  5  │  a9:00.0 │ OFF  2[2,3] NET    (48ch)    │ ON   1[2  ] GDRDMA (48ch)    │
│                 │  6  │  c9:00.0 │ OFF  2[2,3] NET    (48ch)    │ ON   1[3  ] GDRDMA (48ch)    │
│                 │  7  │  e5:00.0 │ ON   1[3  ] GDRDMA (48ch)    │ ON   1[3  ] GDRDMA (48ch)    │
├─────────────────┼─────┼──────────┼──────────────────────────────┼──────────────────────────────┤
│ sendrecv        │  0  │  05:00.0 │ ON   1[0  ] GDRDMA (64ch)    │ ON   1[0  ] GDRDMA (64ch)    │
│                 │  1  │  29:00.0 │ OFF  2[0,1] NET    (64ch)    │ ON   1[0  ] GDRDMA (64ch)    │
│                 │  2  │  49:00.0 │ OFF  2[0,1] NET    (64ch)    │ ON   1[1  ] GDRDMA (64ch)    │
│                 │  3  │  65:00.0 │ ON   1[1  ] GDRDMA (64ch)    │ ON   1[1  ] GDRDMA (64ch)    │
│                 │  4  │  85:00.0 │ ON   1[2  ] GDRDMA (64ch)    │ ON   1[2  ] GDRDMA (64ch)    │
│                 │  5  │  a9:00.0 │ OFF  2[2,3] NET    (64ch)    │ ON   1[2  ] GDRDMA (64ch)    │
│                 │  6  │  c9:00.0 │ OFF  2[2,3] NET    (64ch)    │ ON   1[3  ] GDRDMA (64ch)    │
│                 │  7  │  e5:00.0 │ ON   1[3  ] GDRDMA (64ch)    │ ON   1[3  ] GDRDMA (64ch)    │
└─────────────────┴─────┴──────────┴──────────────────────────────┴──────────────────────────────┘
```

## GDR Coverage Summary

```
┌─────────────────┬────────────────────────────────┬────────────────────────────────┬────────────┐
│ Collective      │ Before-Fix GDR ON              │ After-Fix GDR ON               │ Fix Effect │
├─────────────────┼────────────────────────────────┼────────────────────────────────┼────────────┤
│ all_reduce      │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ alltoall        │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ alltoallv       │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ broadcast       │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ reduce_scatter  │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ all_gather      │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ reduce          │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ scatter         │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ gather          │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ sendrecv        │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
└─────────────────┴────────────────────────────────┴────────────────────────────────┴────────────┘
```

## NIC Assignment per GPU (all_reduce, representative)

```
┌─────┬──────────┬────────────────────────────┬────────────────────────────┐
│ GPU │ GPU BDF  │ Before NICs                │ After NICs                 │
├─────┼──────────┼────────────────────────────┼────────────────────────────┤
│  0  │  05:00.0 │ 1 NIC: ionic_0             │ 1 NIC: ionic_0             │
│  1  │  29:00.0 │ 2 NIC: ionic_0, ionic_2    │ 1 NIC: ionic_0             │ *
│  2  │  49:00.0 │ 2 NIC: ionic_0, ionic_2    │ 1 NIC: ionic_2             │ *
│  3  │  65:00.0 │ 1 NIC: ionic_2             │ 1 NIC: ionic_2             │
│  4  │  85:00.0 │ 1 NIC: ionic_4             │ 1 NIC: ionic_4             │
│  5  │  a9:00.0 │ 2 NIC: ionic_4, ionic_6    │ 1 NIC: ionic_4             │ *
│  6  │  c9:00.0 │ 2 NIC: ionic_4, ionic_6    │ 1 NIC: ionic_6             │ *
│  7  │  e5:00.0 │ 1 NIC: ionic_6             │ 1 NIC: ionic_6             │
└─────┴──────────┴────────────────────────────┴────────────────────────────┘
* = NIC count changed (2 NICs dual-rail -> 1 NIC single-rail with GDR)
```

## Key Observations

1. **Before-fix GDR pattern is CONSISTENT** across all 10 collectives:
   - GDR ON:  GPUs [0, 3, 4, 7] (BDFs: 05:00.0, 65:00.0, 85:00.0, e5:00.0)
   - GDR OFF: GPUs [1, 2, 5, 6] (BDFs: 29:00.0, 49:00.0, a9:00.0, c9:00.0)
   - Coverage: 4/8 = 50%

2. **After-fix GDR pattern is CONSISTENT** across all 10 collectives:
   - GDR ON: ALL 8 GPUs (100%)
   - Coverage: 8/8 = 100%

3. **NIC assignment change** (before -> after):
   - GPU 1 (29:00.0): 2 NIC [ionic_0,ionic_2] (GDR OFF) -> 1 NIC [ionic_0] (GDR ON)
   - GPU 2 (49:00.0): 2 NIC [ionic_0,ionic_2] (GDR OFF) -> 1 NIC [ionic_2] (GDR ON)
   - GPU 5 (a9:00.0): 2 NIC [ionic_4,ionic_6] (GDR OFF) -> 1 NIC [ionic_4] (GDR ON)
   - GPU 6 (c9:00.0): 2 NIC [ionic_4,ionic_6] (GDR OFF) -> 1 NIC [ionic_6] (GDR ON)

### Conclusion

**The DSN mapper fix increases GDR coverage from 4/8 (50%) to 8/8 (100%) across ALL 10 collectives uniformly.**

Every GPU on every collective now uses GDRDMA transport with a single dedicated NIC,
eliminating the mixed GDR-ON/GDR-OFF transport asymmetry that existed before the fix.
