## Stall reasons (stochastic sampling only)

This reference is based on the official documentation, available [here](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/cdna3-cdna4-pc-sampling.html).

Stochastic sampling records a stall reason when `Wave_Issued_Instruction` is 0 (or `wave_issued` is false in JSON). Host-trap sampling does not report stall reasons.

Limitation: PC sampling on CDNA3/CDNA4 focuses on the shader *frontend* — what prevents a wave from *issuing* an instruction. It gives a limited view of execution-pipeline/backend behavior and may not reveal the underlying cause of a backend stall.

| Stall reason | Cause | Analysis / mitigation guidance |
|---|---|---|
| **WAITCNT** | The wave is waiting on a memory dependency (`waitcnt`). | Hotspots often land on `s_waitcnt` after memory ops. High WAITCNT is expected for memory-latency-bound code; use sample counts and source attribution to see *where* the kernel waits, then compare before/after changes. |
| **BARRIER_WAIT** | The wave is waiting at a barrier for other waves in the workgroup to reach the same barrier. | Hotspots at barrier sites show time spent in workgroup synchronization. |
| **ALU_DEPENDENCY** | The instruction could not issue due to an internal hardware dependency (inter-pipeline dependency or data hazard). | See CDNA3/CDNA4 ISA section 4.4 (Data dependency resolution), linked from the doc above. |
| **NO_INSTRUCTION_AVAILABLE** | The wave is stalled waiting for instructions (e.g., at a branch target, I$ miss). | May cluster around control-flow transitions. With host-trap, also account for instruction skid (see Misc. Tips). |
| **INTERNAL_INSTRUCTION** | The wave is issuing an internal instruction (e.g., a `NOP`). | Reflects internal/hardware instructions rather than user kernel logic. |
| **ARBITER_NOT_WIN** | This wave was not selected to issue; multiple waves competed for the same execution pipeline and another wave won. | In JSON, inspect `arb_state_issue_*` for the relevant pipeline. If issue was true, another wave won that cycle and hid the latency of the sampled wave (multi-wave contention). Docs recommend checking for pipeline hotspotting — high contention on one pipe is not always beneficial. |
| **ARBITER_WIN_EX_STALL** | The arbiter selected this wave, but the execution pipeline backpressured it (could not accept more work). | Documented causes: (1) pipeline oversubscription — the pipe hit its limit on outstanding instructions; (2) back-to-back long-latency instructions (e.g., MFMA, vector transcendentals). In JSON, `arb_state_stall_*` == 1 indicates backpressure on that pipeline. |
| **OTHER_WAIT** | Other wait conditions. | Example: wait for XNACK acknowledgment (recoverable page fault). (also described as "other types of wait") |
| **SLEEP_WAIT** | Wave was sleeping. | Reported in the API (`pc_sampling.h`); not listed in the CDNA3/CDNA4 stall table. Treat as an architecture-specific reason if seen. |

**Using arbiter state for deeper diagnosis:** Stochastic JSON samples include `arb_state_issue_*` and `arb_state_stall_*` per execution pipeline (VALU, Matrix, LDS, Scalar, VMEM/Tex, Flat, Exp, Misc). The CDNA3/CDNA4 doc uses these to distinguish pipe latency, SIMD latency, frontend latency, pipeline oversubscription, and pipeline backpressuring — see the "Arbiter state" and "Diagnosing arbiter activity" sections in that file.

**Occupancy context:** `Wave_Count` / `wave_cnt` is the number of active waves on the CU at sample time. The docs note this helps explain how occupancy affects the cost of stalls at a given source or assembly line (but is not a full timeline trace).
