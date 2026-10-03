# Phase 2 — gfx942 SLOT_LIMIT → collectables (AIPROFCOMP-865)

**Branch:** `users/feizheng10/aiprofcomp-865-phase2-weighted-avg`
**Prerequisite:** Phase 1 SPP default (`users/feizheng10/aiprofcomp-865-spp-impl`)

## Summary

All **16** former `SLOT_LIMIT` parents (10 unique PMC sets) are analyze-time
composites. Sub-collectables are single-bucket under SPP; `eval_single_pass_packable`
reports `slot_limit_metrics: 0`.

| # | Unique set | Parents | Composite | Identity |
|---|------------|---------|-----------|----------|
| 1 | VALU FLOPs | `0200.201.0`, `1100.1101.0` | `COLLECT_SUM` F16/F32/F64 rates | Additive rates over shared time |
| 2 | vL1D hit | `0200.201.17`, `0300.301.28`, `1600.1601.0`, `1600.1603.5`, `1600.1603.7` | `COLLECT_SUM` non-RW hit + atom correction | \(100(C-R-W)/C - 100(A_1+A_2)/C\) |
| 3 | HBM Bandwidth | `0400.401.9` | `COLLECT_SUM` rd/wr BW | Additive byte rates |
| 4–7 | AI HBM/L2/L1/LDS | `0400.402.0–3` | `COLLECT_RATIO`(FLOP pieces, bytes) | Per-dispatch \(\sum F / \sum B\) |
| 8 | Perf GFLOPs | `0400.402.4`, `1100.1103.0` | `COLLECT_SUM` VALU+MFMA rates | Additive rates |
| 9 | IPC Issued | `1100.1102.1` | `COLLECT_SUM` two IPC slices | Shared `SQ_ACTIVE_INST_ANY` |
| 10 | Read Instructions | `1500.1504.4` | `COLLECT_SUM` load + (−store−atomic) | TD slot split |

`min`/`max` on composite parents are `None` (avg-only Phase 2) so they do not
re-introduce the full PMC set into grouping.

## Verification

```bash
PYTHONPATH=src:tools python3 tools/eval_single_pass_packable.py --arch gfx942
# expect: slot_limit_metrics: 0, packable_multi: 0, passes: 14, gates PASS
```
