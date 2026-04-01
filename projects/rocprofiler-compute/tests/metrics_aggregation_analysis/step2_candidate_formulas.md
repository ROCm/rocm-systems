# Step 2: Candidate Formulas Derived From Metric Semantics

## Method
For each semantic category from Step 1, we reason through what formula correctly represents the metric's meaning when aggregating across multiple dispatches, using MI3xx hardware counter definitions as ground truth.

---

## Category 1: Throughput (FLOP/s, GB/s, ops/s)

**Pattern**: `AVG(work / time)` where `work` = counter * constant, `time` = End_Timestamp - Start_Timestamp

**What the metric answers**: "What was the sustained throughput across the profiled region?"

**Analysis**:
Consider two dispatches:
- Dispatch 1: 100 FLOPs in 10 ns → 10 GFLOP/s
- Dispatch 2: 100 FLOPs in 1000 ns → 0.1 GFLOP/s

| Formula | Result | Interpretation |
|---------|--------|----------------|
| `AVG(work/time)` | (10 + 0.1)/2 = 5.05 | Average per-dispatch throughput — overweights the short dispatch |
| `SUM(work)/SUM(time)` | 200/1010 = 0.198 | Total work / total time — matches external observation |

An external observer timing the total execution would see 200 FLOPs in 1010 ns. `SUM/SUM` matches this. `AVG` does not.

**Candidate**: `SUM(work) / SUM(time)`

**Rationale**: Throughput is defined as total work divided by total elapsed time. For hardware counters, work counters (SQ_INSTS_VALU_*, TCC_REQ, TCP_TOTAL_CACHE_ACCESSES, etc.) are additive event counts. Time (End_Timestamp - Start_Timestamp) is wall-clock duration per dispatch. SUM(work)/SUM(time) gives the correct total throughput.

**Affected metrics**: VALU FLOPs, VALU IOPs, MFMA FLOPs (all precisions), vL1D/L2/sL1D/L1I Cache BW, L2-Fabric Read/Write BW, Theoretical LDS Bandwidth, LDS LOAD/STORE/ATOMIC Bandwidth, L1I-L2 Bandwidth, sL1D-L2 BW, all "Bandwidth" and "BW" metrics.

---

## Category 2: Utilization (busy_cycles / total_cycles as %)

**Pattern**: `AVG(100 * BUSY / ($GRBM_GUI_ACTIVE_PER_XCD * $cu_per_gpu))`

**What the metric answers**: "What fraction of available hardware cycles was this unit busy?"

**Analysis**:
The denominator contains `$GRBM_GUI_ACTIVE_PER_XCD` (per-dispatch GPU active cycles — from HW: "The GUI is Active" cycle count) multiplied by `$cu_per_gpu` (hardware constant).

Consider two dispatches:
- Dispatch 1 (short kernel): VALU busy 50 out of 100 CU-cycles → 50%
- Dispatch 2 (long kernel): VALU busy 900 out of 10000 CU-cycles → 9%

| Formula | Result | Interpretation |
|---------|--------|----------------|
| `AVG(busy/total)` | (50% + 9%)/2 = 29.5% | Equal weight per dispatch — short kernel dominates |
| `SUM(busy)/SUM(total)` | 950/10100 = 9.4% | Time-weighted — overall fraction of CU-cycles that were busy |

The question "what fraction of total CU-cycles was VALU busy?" is answered by 950/10100 = 9.4%. The short dispatch consumed only 1% of total CU-cycles but contributes 50% to the AVG result. SUM/SUM reflects the actual hardware utilization.

**Candidate**: `(100 * SUM(BUSY)) / (SUM($GRBM_GUI_ACTIVE_PER_XCD) * $cu_per_gpu)`

Note: `$cu_per_gpu` is a hardware constant, so it factors out of SUM: `SUM(BUSY * 100) / ($cu_per_gpu * SUM(GRBM_ACTIVE))`.

**Rationale**: Utilization measures fraction of available cycles used. When aggregating across dispatches, total busy cycles / total available cycles gives the time-weighted utilization, which correctly reflects how much of the hardware's capacity was consumed.

**Affected metrics**: SALU/VALU/MFMA/VMEM/Branch/CU Utilization, L2 Utilization, vL1D Utilization, LDS Utilization/Access Rate, Address Processing Unit Busy, Data-Return Busy, Accelerator/Scheduler-Pipe/Workgroup Manager/Shader Engine/SIMD Utilization, CPF/CPC Utilization, all stall-as-percent-of-busy metrics (Read Stall, Write Stall, L2-Fabric stalls, CPF/CPC stalls).

---

## Category 3: Rate (hits/total, events_A/events_B as %)

**Pattern**: `AVG(100 * A / B)` where A ⊂ B (A is a subset of B, or A and B are related event counts)

**What the metric answers**: "What fraction of total events were of type A?"

**Analysis** (the bug report's exact example):
- Dispatch 1: 2 hits out of 300 accesses → 0.67%
- Dispatch 2: 90 hits out of 100 accesses → 90%

| Formula | Result | Interpretation |
|---------|--------|----------------|
| `AVG(A/B)` | (0.67% + 90%)/2 = 45.3% | Average per-dispatch rate — misleading |
| `SUM(A)/SUM(B)` | 92/400 = 23% | Overall rate across all events |

"What fraction of all accesses were hits?" → 92 out of 400 = 23%. SUM/SUM.

**Candidate**: `(100 * SUM(A)) / SUM(B)`

**Rationale**: For rates/percentages derived from event counts (hits, misses, stalls, traffic fractions), the correct aggregate is total events of type A / total events. Each individual event contributes equally. HW counters (TCC_HIT, TCC_MISS, TCC_REQ, etc.) are simple event counts — additive by nature.

**Affected metrics**: L2/vL1D/sL1D/L1I Cache Hit Rate, UTCL1 Hit Ratio, HBM/Remote/Uncached Read/Write Traffic %, Atomic Traffic %, Bank Conflict Rate, Coalescing, vL1D stall rates (Stalled on L2 Data, etc.), VALU Co-Issue Efficiency.

---

## Category 4: Latency (accumulated_level / request_count)

**Pattern**: `AVG(LEVEL_sum / REQ_sum)` where LEVEL is a hardware level counter

**What the metric answers**: "On average, how many cycles did each request/instruction take?"

**Hardware mechanism** (from MI350 perfmon spec): Level counters measure the number of in-flight events at each sampling instant. `SQ_ACCUM_PREV_HIRES` is a hardware accumulator that integrates the level counter over time: `I(T) = ∫L(t)dt`. The HW spec explicitly states: *"Average read latency = TCC_PERF_SEL_EA_RDREQ_LEVEL / TCC_PERF_SEL_EA_RDREQ"*.

**Analysis**:
- Dispatch 1: LEVEL=500, REQ=100 → 5 cycles/req (light workload, 100 requests)
- Dispatch 2: LEVEL=6000, REQ=1000 → 6 cycles/req (heavy workload, 1000 requests)

| Formula | Result | Interpretation |
|---------|--------|----------------|
| `AVG(LEVEL/REQ)` | (5 + 6)/2 = 5.5 | Average per-dispatch latency — each dispatch weighted equally regardless of request volume |
| `SUM(LEVEL)/SUM(REQ)` | 6500/1100 = 5.91 | Per-request weighted average — each request contributes equally |

**Both are valid statistics measuring different things**:
- `SUM(LEVEL)/SUM(REQ)`: "What was the average latency experienced by a randomly chosen request across the entire workload?" (request-weighted)
- `AVG(LEVEL/REQ)`: "What was the average per-dispatch latency?" (dispatch-weighted — a dispatch with 10 requests weighs the same as one with 10000)

The `3000_mem_bw.yaml` reference file provides both side-by-side:
- "L1 average latency" using `SUM/SUM`: *"Weighted average across all CUs (SUM/SUM gives more weight to CUs with more accesses)"*
- "L1 latency" using `AVG/AVG`: *"Ratio of averages across CUs (AVG/AVG gives equal weight to each CU)"*

**Candidate 1 (primary)**: `SUM(LEVEL_sum) / SUM(REQ_sum)` — request-weighted average latency
**Candidate 2 (secondary)**: `AVG(LEVEL_sum) / AVG(REQ_sum)` — dispatch-weighted average latency

Note on candidate 2: `AVG(A)/AVG(B)` is mathematically equivalent to `SUM(A)/SUM(B)` when there are no NaN mismatches between A and B (same dispatch count for both). With NaN mismatches: `AVG(A)` = `SUM(A)/count_nonnan_A`, `AVG(B)` = `SUM(B)/count_nonnan_B`, so `AVG(A)/AVG(B)` = `(SUM(A)*count_B)/(SUM(B)*count_A)` — different from `SUM(A)/SUM(B)`. This needs empirical investigation in Step 4.

Note on current formula: `AVG(LEVEL/REQ)` is the *least* defensible of the three options because it overweights dispatches with few requests where a single slow request can produce an outlier ratio. The HW was designed for LEVEL/REQ as a division of totals, not as a per-dispatch ratio.

**Affected metrics**: L2-Fabric Read/Write/Atomic Latency, VMEM/SMEM/LDS Latency, L1I Fetch Latency, MFMA Instruction Cycles, VL1 Lat, L2 Rd/Wr Lat, Fabric Rd/Wr/Atomic Lat, SPI-SQ Wave Allocated, VGPR/SGPR Writes (cycles/wave).

---

## Category 5: Normalized Counters (COUNTER / $denom)

**Pattern**: `AVG(COUNTER / $denom)` where `$denom` depends on normalization mode

**What the metric answers**: "How many events per normalization unit?"

**$denom types and their behavior**:

| $denom mode | $denom value | $denom nature | AVG vs SUM/SUM |
|-------------|-------------|---------------|----------------|
| per_kernel | 1 | constant | `AVG(C/1) = AVG(C) = SUM(C)/N` — identical |
| per_wave | SQ_WAVES | per-dispatch counter | `AVG(C/WAVES)` ≠ `SUM(C)/SUM(WAVES)` — same as rate |
| per_cycle | $GRBM_GUI_ACTIVE | per-dispatch counter | `AVG(C/ACTIVE)` ≠ `SUM(C)/SUM(ACTIVE)` — same as utilization |
| per_second | (End_Timestamp-Start_Timestamp)/1e9 | per-dispatch time | `AVG(C/time)` ≠ `SUM(C)/SUM(time)` — same as throughput |

**For per_kernel ($denom=1)**: No difference. Both give `AVG(COUNTER)`.

**For all other modes**: The analysis is identical to throughput/rate/utilization — `SUM(COUNTER)/SUM($denom)` gives the overall normalized count, while `AVG(COUNTER/$denom)` gives dispatch-weighted average.

**Candidate**: `SUM(COUNTER) / SUM($denom)`

**Rationale**: For per_wave: "how many instructions per wavefront overall?" = total instructions / total wavefronts. For per_cycle: "how many events per GPU active cycle overall?" = total events / total active cycles. For per_second: same as throughput. In all cases, SUM/SUM gives the overall rate.

**Exception**: For `per_kernel` mode, `AVG(COUNTER)` is already correct (no ratio involved).

**Affected metrics**: All `COUNTER / $denom` metrics in L2 Cache (Req, Hits, Misses, Writeback, etc.), vL1D (Total Req, Read/Write/Atomic Req, Cache Accesses, etc.), LDS (Instructions, Index Accesses, Bank Conflict cycles, etc.), Instruction Mix (all instruction counts), TA/TD instruction counts, Wavefront stats (Wave Cycles, Wait Cycles, Active Cycles), L2 stall cycles, UTCL1 stall cycles. ~100+ metrics across all panels.

---

## Category 6: Compound Metrics (ratio of two ratios, or scaling by constant and $denom)

**Pattern**: `AVG((100 * COUNTER) / ($GRBM_GUI_ACTIVE_PER_XCD * $cu_per_gpu))` where `$cu_per_gpu` is a hardware constant

**Analysis**: When the denominator mixes a per-dispatch counter with a hardware constant:
- Current: `AVG(100 * A / (ACTIVE * CU_COUNT))`
- Proposed: `(100 * SUM(A)) / (SUM(ACTIVE) * CU_COUNT)`

The constant `CU_COUNT` factors out of SUM. This is algebraically correct.

For formulas with nested division like `AVG(((100 * A) / ACTIVE) / CU_COUNT)`:
- This is mathematically identical to `AVG(100 * A / (ACTIVE * CU_COUNT))`
- Candidate: `(100 * SUM(A)) / (SUM(ACTIVE) * CU_COUNT)` — same result

**Candidate**: Factor constants out of SUM; keep per-dispatch counters inside SUM.

---

## Category 7: IPC and Per-Busy-Cycle Ratios

**Pattern**: `AVG(SQ_INSTS / SQ_BUSY_CU_CYCLES)`

**What the metric answers**: "How many instructions were issued per busy CU cycle?"

**Analysis**: Same as normalized counters with `per_cycle` denominator. `SUM(INSTS)/SUM(BUSY_CYCLES)` gives the overall IPC. This weights busy dispatches (those with more busy cycles) more heavily, which is correct — a dispatch that kept the CU busy for 10000 cycles contributes more to the overall IPC than one that was busy for 100 cycles.

**Candidate**: `SUM(SQ_INSTS) / SUM(SQ_BUSY_CU_CYCLES)`

---

## Category 8: VALU Active Threads

**Pattern**: `AVG(SQ_THREAD_CYCLES_VALU / SQ_ACTIVE_INST_VALU)`

**HW definition**: `SQ_THREAD_CYCLES_VALU` = "quad-cycles VALU is in use times number of active threads". This is already a weighted counter — it accumulates (cycles * thread_count) per cycle.

**What the metric answers**: "On average, how many work-items were active per VALU instruction?"

**Candidate**: `SUM(SQ_THREAD_CYCLES_VALU) / SUM(SQ_ACTIVE_INST_VALU)` — instruction-weighted average divergence

**Rationale**: Each VALU instruction-cycle contributes equally to the overall divergence measure. Dispatches with more VALU work should dominate.

---

## Category 9: MIN/MAX of Ratios

**Pattern**: `MIN(A/B)`, `MAX(A/B)` in the `min:` and `max:` YAML fields

**What these show**: The minimum/maximum per-dispatch value of the ratio.

**Analysis**: MIN and MAX are inherently per-dispatch statistics — they show extremes. There is no "weighted MIN" that makes mathematical sense. The question is whether per-dispatch extremes of ratios are meaningful:

- `MIN(hit_rate)` = "lowest per-dispatch hit rate" — meaningful for identifying worst-case dispatches
- `MAX(latency)` = "highest per-dispatch latency" — meaningful for identifying latency spikes
- `MIN(throughput)` = "lowest per-dispatch throughput" — meaningful but could be skewed by very short dispatches

**Candidate**: Keep `MIN(A/B)` and `MAX(A/B)` as-is. These are per-dispatch distribution statistics, not aggregates. They are valid as long as users understand they show dispatch-level extremes.

**Caveat for empirical validation**: Verify with profiled data whether MIN/MAX of ratios produces misleading outliers (e.g., a dispatch with 1 request and anomalous latency dominating MAX).

---

## Summary: Candidate Formula Per Category

| Category | Current | Candidate 1 (Primary) | Candidate 2 (if any) |
|----------|---------|----------------------|---------------------|
| Throughput | `AVG(work/time)` | `SUM(work)/SUM(time)` | — |
| Utilization | `AVG(100*busy/total)` | `(100*SUM(busy))/SUM(total)` | — |
| Rate | `AVG(100*A/B)` | `(100*SUM(A))/SUM(B)` | — |
| Latency | `AVG(level/req)` | `SUM(level)/SUM(req)` | `AVG(level)/AVG(req)` (if NaN behavior differs) |
| Normalized (per_wave/per_cycle/per_second) | `AVG(counter/$denom)` | `SUM(counter)/SUM($denom)` | — |
| Normalized (per_kernel) | `AVG(counter)` | No change needed | — |
| IPC | `AVG(insts/cycles)` | `SUM(insts)/SUM(cycles)` | — |
| Active Threads | `AVG(thread_cycles/active_inst)` | `SUM(thread_cycles)/SUM(active_inst)` | — |
| MIN/MAX | `MIN(A/B)`, `MAX(A/B)` | No change (per-dispatch extremes) | — |
