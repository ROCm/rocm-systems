# Timing Model

rocjitsu executes GPU code correctly but, by default, in no particular time: a
kernel takes as long as the host needs to emulate it, and every guest-visible
clock reports the host's. The timing plane replaces that with a modelled
duration, produced by a set of clocked simdojo components that observe what the
functional simulator actually did.

It is off unless a config asks for it. With it off, the only cost is a null
check on the paths that would have fed it.

## What it is for

Two things, and they pull in different directions.

The first is that a **guest profiler should measure the model.** `hipEventElapsedTime`,
`torch.cuda.Event`, the KFD clock counters, a completion signal's timestamps and
`s_memtime` inside a kernel all resolve to the modelled clock, so an unmodified
PyTorch profile taken under emulation reports modelled microseconds rather than
emulator wall clock. Nothing in the guest is aware of this.

The second is that the model should be **wrong in a knowable direction.** Every
parameter it cannot read from the config resolves to the slowest reasonable
value for that parameter, and the run reports which ones it had to guess. An
incomplete config makes a run look slow and suspicious; it never makes one look
fast and accurate.

## Shape

```
functional simulator                timing plane
────────────────────                ────────────
command processor  ──dispatch──▶    DispatchDes      composes the duration
compute unit       ──wavefront─▶    ComputeUnitDes   issue, scoreboard, stalls
                   ──instruction▶       │
                                        ├──▶ CacheDes    L1V / L1S / L1I / L2 / MALL
                                        └──▶ ChannelDes  fabric, DRAM
                                             │
                                    TimingEngine       one min-heap over ticks
                                             │
                                    SimulatedClock ──▶ every guest-visible clock
```

`TimingCollector` is the adapter between the two sides. It classifies each
retired instruction, recovers the registers it named and the addresses its lanes
computed, and hands a `RetiredInstruction` to the compute unit that ran it. It
holds no timing state of its own.

| File | What lives there |
|------|------------------|
| `timing/tuning.{h,cpp}` | Every parameter, and the ledger of which ones the config named |
| `timing/inst_class.h` | Instruction classes and functional units, at the granularity timing cares about |
| `timing/classify.{h,cpp}` | Opcode to class |
| `timing/coalesce.h` | Lane addresses to cache lines; local-data-share bank conflicts |
| `timing/timed_component.{h,cpp}` | Base for a component with an inbox and a clock domain |
| `timing/engine.{h,cpp}` | The event loop the plane runs on |
| `timing/cu_des.{h,cpp}` | Compute unit: issue, register scoreboard, wait counters, memory stalls |
| `timing/cache_des.{h,cpp}` | A cache level and its tag array |
| `timing/dram_des.{h,cpp}` | A bandwidth-limited channel |
| `timing/dispatch_des.{h,cpp}` | Composition, and the per-dispatch bandwidth ledger |
| `timing/timing_plane.{h,cpp}` | Wiring, routing, and the device clock |
| `timing/collector.{h,cpp}` | The adapter from the functional simulator |
| `timing/simulated_clock.{h,cpp}` | What the guest reads |

## What a dispatch costs

```
cycles = launch + max(issue, bandwidth, placement) + latency + ramp
```

- **issue** — the busiest port's queue on the busiest compute unit. Each
  instruction occupies its functional unit for its class's issue cycles, and
  the unit retires that queue at its port count. Every instruction *also*
  occupies the front end, whatever unit it then goes to.
- **bandwidth** — the busiest instance of the busiest level: first-level cache,
  second-level cache, fabric, memory-side cache, DRAM. Charged from the lines
  the access actually touched.
- **placement** — the command processor can only start so many workgroups per
  cycle.
- **latency** — the longest wavefront's dependence chain, less whatever the
  other resident wavefronts had to issue while it waited, plus one unavoidable
  round trip to the deepest level the dispatch's traffic reached. A dispatch
  with enough parallelism to hide its latency has an exposed term near zero and
  is purely throughput bound; one with none has a throughput term near zero and
  is purely its own critical path.
- **ramp** — filling the machine is serial and precedes steady state.

Throughput and latency **add** rather than taking the larger. A resource has to
fill before it can drain: the first access of a streaming kernel still waits a
full round trip before any byte arrives, and only then does the bandwidth term
describe what follows.

### Why the addresses matter

Coalescing runs on the addresses the kernel *computed*, not on addresses
inferred from source. That distinction is the single largest error source in
published analytical GPU models: a wave64 dword load is two cache lines when
contiguous and sixty-four when divergent, a thirtyfold difference that no static
analysis recovers. The same is true of local-data-share bank conflicts, which
are resolved per phase group from the real addresses.

This is the one thing a model inside a functional simulator can do that a model
outside one cannot, and most of the accuracy comes from it.

## Making a component clocked

The plane is ordinary simdojo. To give a functional component a clocked
counterpart:

1. Derive from `TimedComponent`, which supplies an inbox, a clock domain and
   `advance(now)`.
2. Implement `advance()`: drain the inbox, `reserve(ready, cycles)` on whatever
   the component serves, and hand the result downstream or to the completion
   callback.
3. Construct it in `TimingPlane` and wire it, and add its parameters to
   `Tuning`.

`CacheDes` is the smallest complete example: it probes a tag array, splits a
request into the part it can serve and the part it cannot, and forwards one
message rather than one per line.

## Configuration

Everything is read from one `timing` block in the config rocjitsu is already
given. There is no second file and no search path, so a run's timing is
reproducible from one artefact.

```json
{
  "timing": {
    "enabled": true,
    "clock_mhz": 2100,
    "machine": { "compute_units": 256, "vector_alu.issue_cycles": 1, "...": "..." }
  }
}
```

The block is parsed in a second, schema-free pass over the same document. The
typed config load runs with unexpected fields skipped, so a timing block on the
typed path would be dropped in silence and the plane would run entirely on
fallbacks with nothing to say it had.

mirage writes this block from its agent table; see `mirage/docs/timing.md`.
`--timing-tuning PATH` overlays a separate file, which is where calibrated
values live: they are measurements against hardware rather than properties
derivable from a datasheet, and they are not in this repository.

### Two kinds of number

The config mixes two things and names them apart on purpose.

**Machine parameters** describe the part: how many compute units, how wide a
cache line is, how many cycles a matrix instruction occupies its pipe. A reader
can check these against a datasheet.

**Calibration** is fitted to measurements: `stall_exposed_fraction`,
`latency_exposure_scale`, `fill_exposure_scale`, `fill_ramp_scale`,
`straggler_cycles`, the per-class `front_end.*.issue_cycles`, and the fractional
port counts. Each is there because a term is right in shape and wrong in size
for a reason the model cannot presently name. They default to neutral, so a
config without them behaves as the uncalibrated model.

## Reporting

| Variable | What it does |
|----------|--------------|
| `ROCJITSU_TIMING_REPORT` | Writes an end-of-run report: per-dispatch durations, which term bound each, cache and channel totals, and the resolved/fell-back parameter ledger |
| `ROCJITSU_TIMING_TRACE` | Writes one JSON line per dispatch with every term the composition used |

The trace exists so that a tuning question can be answered without re-running
the corpus. `tests/corpus/meter_refit.py` joins it to the meter's per-case
device-clock windows and recomputes the whole corpus offline in about a second,
against ten minutes for a real pass. It reproduces the emulator's own answer to
within a fraction of a per cent, and `--verify` reports that drift on every run
so that a mistake in the offline model cannot be read as an improvement.

## Accuracy

Validated against `tests/corpus/rocm-meter.py` run under the emulator and scored
by `tests/corpus/meter_score.py` against a recorded gfx950 reference. See
`tests/corpus/README.md` for how to reproduce, and `Testing/reports/` for the
generated report — which is not checked in.

## Testing

`tests/timing/timing_model_test.cpp` runs in CI. Every assertion is about a
*relation* that holds whatever the tuning says, never about a number the tuning
supplies: more work costs more, the same input gives the same answer, a
divergent access costs more than a coalesced one, an unnamed parameter makes a
run slower rather than faster. That is what lets the tests live here while the
calibrated values do not.
