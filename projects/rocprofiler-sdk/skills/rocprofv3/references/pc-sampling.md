# PC sampling with rocprofv3 (beta)

Contents:

1. [Methods and supported GPUs](#methods-and-supported-gpus)
2. [Checking the configuration](#checking-the-configuration)
3. [Collecting samples](#collecting-samples)
4. [Output](#output)
5. [Quick analysis without the PC sampling skill](#quick-analysis-without-the-pc-sampling-skill)
6. [Stall reasons](#stall-reasons)
7. [Arbiter state](#arbiter-state)
8. [Troubleshooting](#troubleshooting)

PC sampling periodically records the program counter of running waves, which shows where a
kernel spends its time across many dispatches. For full analysis, the rocprofiler-sdk PC
sampling skill ships `analyze_pc_sampling.py`; use it when it is installed. Sources: the
rocprofiler-sdk PC sampling guides (`using-pc-sampling`, `cdna3-cdna4-pc-sampling`) and the
rocprofiler-sdk PC sampling skill. Not measured for this skill: on the MI325X used to validate the
other modes, both sampling methods hung the profiled application.

## Methods and supported GPUs

| Method | GPUs | Unit | Notes |
| --- | --- | --- | --- |
| `stochastic` | MI300A, MI300X, MI325X (gfx942), MI350X, MI355X (gfx950) | `cycles`, power of two | Hardware sampling, no skid, reports whether the wave issued and why not (stall reason) |
| `host_trap` | MI200 (gfx90a) and later | `time` (microseconds) | Software interrupt; samples can skid up to two instructions; no stall reasons |

PC sampling is documented only for the Instinct GPUs above; MI100 lacks the hardware support.
Prefer `stochastic` wherever it is available.
On MI300X, ROCm 7.0 fixed both methods; host trap needs PSP TOS firmware 0x00360259 or later and
stochastic needs MEC firmware feature version 50 with firmware 0x1a or later
(`sudo cat /sys/kernel/debug/dri/<N>/amdgpu_firmware_info | grep -E 'SOS|MEC'`).

## Checking the configuration

```bash
rocprofv3-avail info --pc-sampling     # or: rocprofv3 -L
```

Each GPU lists its methods with `Unit`, `Min_Interval`, `Max_Interval`, and `Flags`. On gfx942
with rocprofv3 1.4.1: `host_trap`, unit time, 1 to 2^64-1; `stochastic`, unit cycle, 256 to
1048576, flag `interval pow2`. The command must match a listed configuration exactly. Only one PC
sampling configuration can be active on a GPU at a time: if a GPU lists only one configuration,
another process is probably sampling it already.

## Collecting samples

Build with `-g` and the usual optimization so samples map to source lines.

```bash
# MI300 and later
rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method stochastic --pc-sampling-unit cycles \
  --pc-sampling-interval 1048576 --kernel-trace --output-format json -d rocprof_pcs -o run -- ./app

# MI200
rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method host_trap --pc-sampling-unit time \
  --pc-sampling-interval 1000 --kernel-trace --output-format json -d rocprof_pcs -o run -- ./app
```

- All four `--pc-sampling-*` options are required; `ROCPROFILER_PC_SAMPLING_BETA_ENABLED=ON` is
  the environment equivalent of the first.
- Add `--kernel-trace` so samples can be attributed to kernel names.
- Use `--output-format json` (richest, required for stall details) or `csv`. The default rocpd
  output does not carry PC samples.
- Intervals: stochastic 2^18 to 2^20 cycles (larger is cheaper; smaller intervals produce huge
  outputs quickly); host trap about 1000 us, lowered if a short workload yields few samples.
- PC sampling cannot be combined with `--spm` or `--replay-mode kernel`. `--selected-regions`
  limits it to code between `roctxProfilerResume(0)` and `roctxProfilerPause(0)`; the kernel
  filter options do not apply to it.

## Output

- JSON: `<dir>/<name>_results.json`. Samples are under
  `rocprofiler-sdk-tool[0].buffer_records.pc_sample_stochastic` (or `pc_sample_host_trap`); each
  has `record` (`dispatch_id`, `pc.code_object_id`, `pc.code_object_offset`, `exec_mask`,
  `wave_cnt`; stochastic adds `wave_issued`, `inst_type`, and `snapshot` with `stall_reason` and
  `arb_state_issue_*`/`arb_state_stall_*`) and `inst_index`, an index into
  `strings.pc_sample_instructions` (ISA text) and `strings.pc_sample_comments` (source line when
  built with `-g`). Kernel names come from `kernel_symbols` and `buffer_records.kernel_dispatch`.
- CSV: `*_pc_sampling_stochastic.csv` or `*_pc_sampling_host_trap.csv` with `Dispatch_Id`,
  `Instruction`, `Instruction_Comment`, `Exec_Mask`, `Sample_Timestamp`, and for stochastic
  `Wave_Issued_Instruction`, `Instruction_Type`, `Stall_Reason`, `Wave_Count`; pair it with
  `*_kernel_trace.csv` for kernel names.

## Quick analysis without the PC sampling skill

Top sampled instructions and stall reasons from the JSON (standard library only):

```python
import collections, json, sys

tool = json.load(open(sys.argv[1]))["rocprofiler-sdk-tool"][0]
strings, records = tool["strings"], tool["buffer_records"]
samples = records.get("pc_sample_stochastic") or records.get("pc_sample_host_trap") or []
insts, comments = strings.get("pc_sample_instructions", []), strings.get("pc_sample_comments", [])
by_inst, stalls = collections.Counter(), collections.Counter()
for s in samples:
    by_inst[s.get("inst_index", -1)] += 1
    if s["record"].get("wave_issued") == 0:
        stalls[s["record"].get("snapshot", {}).get("stall_reason", "").split("_REASON_")[-1]] += 1
print(f"{len(samples)} samples")
for idx, n in by_inst.most_common(15):
    text = insts[idx] if 0 <= idx < len(insts) else "<unknown code object>"
    src = comments[idx] if 0 <= idx < len(comments) else ""
    print(f"{n:8d} {100 * n / len(samples):5.1f}%  {text:45.45s} {src}")
for reason, n in stalls.most_common():
    print(f"stall {reason}: {n}")
```

Check that the run produced a meaningful number of samples (thousands, not tens) before drawing
conclusions; tiny workloads need a smaller interval or more iterations.

## Stall reasons

Stochastic sampling records a stall reason when the sampled wave did not issue. It describes the
shader frontend (why a wave could not issue), not the execution pipelines' internal causes.

| Stall reason | Meaning | What to look at |
| --- | --- | --- |
| `WAITCNT` | Waiting on a memory dependency (`s_waitcnt`) | Expected in memory-bound code; check coalescing and bytes moved with counter collection |
| `BARRIER_WAIT` | Waiting at a workgroup barrier | Load imbalance between waves, too many barriers |
| `ALU_DEPENDENCY` | Data hazard or inter-pipeline dependency | Dependency chains; see the CDNA3/CDNA4 ISA section on data dependency resolution |
| `NO_INSTRUCTION_AVAILABLE` | No instruction ready: branch target, instruction cache miss | Control-flow transitions, large unrolled code |
| `INTERNAL_INSTRUCTION` | Issuing an internal instruction such as `s_nop` | Hardware hazards, not user logic |
| `ARBITER_NOT_WIN` | Another wave won the pipeline this cycle | Contention on one pipeline; occupancy may hide it |
| `ARBITER_WIN_EX_STALL` | Selected, but the pipeline back-pressured | Oversubscribed pipeline or back-to-back long-latency instructions (MFMA, transcendentals) |
| `OTHER_WAIT` | Other waits, for example XNACK page-fault acknowledgment | Unified or managed memory faults |
| `SLEEP_WAIT` | Wave sleeping (`s_sleep`) | Spin-wait loops |

`wave_cnt` (CSV `Wave_Count`) is the number of active waves on the CU at sample time, which shows
how occupancy evolves and how much latency other waves can hide.

## Arbiter state

Stochastic JSON samples carry `arb_state_issue_PIPE` and `arb_state_stall_PIPE` for the VALU,
Matrix, LDS, Scalar, VMEM/Tex, Flat, Exp, and Misc pipelines:

- `issue == 1 and stall == 0`: the pipe accepted an instruction this cycle. Counting such pipes
  estimates instructions per cycle (count VALU twice when `dual_issue_valu == 1`).
- A wave stalled on `ARBITER_NOT_WIN` while its pipe has `issue == 1 and stall == 0`: pipeline
  oversubscription by several waves.
- `stall == 1` with the wave stalled on `ARBITER_WIN_EX_STALL`: pipeline back-pressure.
- `issue == 0 or stall == 1` on a pipe: no forward progress on that pipe; on every pipe at once,
  the whole SIMD is stalled.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `PC sampling unavailable. The feature is implicitly disabled` | Add `--pc-sampling-beta-enabled`. |
| `All three PC sampling configurations need to be set` | Pass method, unit, and interval together. |
| Configuration rejected | Match `rocprofv3-avail info --pc-sampling` exactly; stochastic intervals must be powers of two. |
| Only one configuration listed, or sampling fails to start | Another process is sampling that GPU; wait for it or use another GPU with `HIP_VISIBLE_DEVICES`. |
| Application hangs or runs orders of magnitude slower | Stop it (if Ctrl+C does not end it, kill it; that run's data is lost), check the firmware versions above, try the other method or a larger interval, and fall back to thread trace for instruction-level data. |
| Few samples | Larger workload, more iterations, or a smaller interval. |
| Empty source column | Rebuild with `-g`. |
| Hotspot looks shifted by one or two instructions (host trap) | Skid: inspect neighboring instructions and branch targets, or use stochastic. |
