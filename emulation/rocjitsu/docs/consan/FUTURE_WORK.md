# ConSan future work

[STATUS_RDNA4.md](STATUS_RDNA4.md) records the gfx1201 north-star matrix.  The
2026-07-22 Record/Replay audit repaired clean-input regressions exposed after
the gfx1250-focused work: hardware dispatch identity is now used on RDNA4, and
RDNA4 relay sizing no longer applies a large-object envelope indiscriminately.
All 19 clean and paired-overhead rows pass at one source-matched tip.  The same
audit corrected two overstated fault claims: TP2 detects its exact drop in 3/5
trials and D128 pressure in 0/5.  The bounded 4-dispatch × 4-owner candidate
now restores D128-pressure detection to 5/5 while remaining clean in 5/5
trials.  TP2 cannot be rerun in the current workspace because its IREE test
suite and build are absent, so source-matched TP2 and overhead/memory evidence
remain the active release-certificate gap.

Prioritize demonstrable end-to-end LLM value over isolated-kernel breadth. Do
not reopen a completed cell merely to accumulate more worklog. If new evidence
finds a regression, update `STATUS_RDNA4.md` immediately and make that repair
the highest-priority runnable node.

Every solid edge means a real prerequisite: `A --> B` means finish `A` before
attempting `B`. Dashed edges denote unavailable-hardware dependencies.

## Status legend

```mermaid
flowchart LR
  D["DONE"]:::done
  A["NEXT"]:::active
  T["FUTURE"]:::todo
  G{"CHECKPOINT"}:::target
  X["HARDWARE-DEFERRED"]:::deferred

  classDef done fill:#70ad47,stroke:#274e13,stroke-width:2px,color:#000;
  classDef active fill:#ffc000,stroke:#7f6000,stroke-width:3px,color:#000;
  classDef todo fill:#a5a5a5,stroke:#404040,stroke-width:2px,color:#000;
  classDef target fill:#ed7d31,stroke:#843c0c,stroke-width:3px,color:#000;
  classDef deferred fill:#9e7cc1,stroke:#351c75,stroke-width:2px,color:#000;
```

## Program overview

```mermaid
flowchart LR
  S["RDNA4 Record/Replay clean, coverage and<br/>overhead requalified at one tip"]:::done
  R["R ACTIVE: restore dispatch-coherent<br/>Record/Replay fault evidence"]:::active
  D["D: stronger real-world<br/>fault detection"]:::todo
  H["H: adversarial-input and<br/>metadata hardening"]:::todo
  B["B: admit additional<br/>real workloads"]:::todo
  P["P: qualify another<br/>gfx architecture"]:::deferred

  S --> R
  S --> D
  S --> H
  S --> B
  R -. requires target hardware .-> P

  classDef done fill:#70ad47,stroke:#274e13,stroke-width:2px,color:#000;
  classDef active fill:#ffc000,stroke:#7f6000,stroke-width:3px,color:#000;
  classDef todo fill:#a5a5a5,stroke:#404040,stroke-width:2px,color:#000;
  classDef deferred fill:#9e7cc1,stroke:#351c75,stroke-width:2px,color:#000;
```

The four RDNA4 tracks are technically independent.  `R` is front-loaded
because new evidence invalidated two Record/Replay greens.  Work on `D`, `H`,
or `B` should rerun the highest-value affected e2e sentinel immediately.

## R: restore the Record/Replay release certificate

This track preserves the corrected generation semantics shared with gfx1250.
It must not regain RDNA4 detections by comparing different dispatches, hiding
an expert knob in ordinary operation, or enabling unbounded dynamic append.

```mermaid
flowchart TB
  R0["R0 DONE: rebuild source-matched hook;<br/>audit all 19 Record/Replay rows"]:::done
  R1["R1 DONE: fix RDNA4 dispatch-ID state<br/>and selective relay envelopes"]:::done
  R2["R2 DONE: 19/19 clean and 57/57 paired<br/>overhead rows; repeat TP2 clean 5/5"]:::done
  R3["R3 DONE: exact faults expose TP2 3/5<br/>and D128 pressure 0/5 detection"]:::done
  R4["R4 DONE: bound capture to four dispatch<br/>buckets × four owner slots"]:::done
  R5["R5 DONE: model, emission and host tests;<br/>gfx1250/gfx942/gfx950 gates pass"]:::done
  R6["R6 ACTIVE: D128 clean/fault passes;<br/>rerun TP2 and frozen overhead/memory"]:::active
  R7["R7: publish source-matched evidence and<br/>restore only evidence-backed greens"]:::todo
  RG{"gfx1201 RECORD/REPLAY<br/>CERTIFICATE RESTORED"}:::target

  R0 --> R1 --> R2 --> R3 --> R4 --> R5 --> R6 --> R7 --> RG

  classDef done fill:#70ad47,stroke:#274e13,stroke-width:2px,color:#000;
  classDef active fill:#ffc000,stroke:#7f6000,stroke-width:3px,color:#000;
  classDef todo fill:#a5a5a5,stroke:#404040,stroke-width:2px,color:#000;
  classDef target fill:#ed7d31,stroke:#843c0c,stroke-width:3px,color:#000;
```

Automatic layouts deliberately expand each logical range from the historical
single 64-byte record to a four hardware-dispatch × four canonical-owner grid
of 72-byte records: 1,152 bytes per range, an 18× static access-storage
increase.  The 128 MiB per-buffer ceiling therefore rejects some inventories
that previously fit; planner boundary tests pin that tradeoff.  A reversible
64-bit dispatch claim owns each outer bucket.  A different dispatch colliding
with that bucket, a different workgroup reusing a dispatch/owner slot, an owner
outside the bounded set, or an unrepresentable/in-flight claim emits typed
Record/Replay saturation and makes dynamic evidence incomplete.  Direct
size-derived caller layouts remain 1 × 1, but now apply the same exact
dispatch/workgroup reuse qualification.  This is a bounded observation policy,
not an exhaustive trace or a workload-specific control.

D128-pressure validates the intended behavior on physical gfx1201: five clean
trials are complete and false-positive-free, and all five exact-drop trials
diagnose the fault while the oracle fails.  Acceptance still requires TP2 plus
paired overhead and memory on a clean committed tree with a hook rebuilt from
exactly that tree.  Run no more than four GPU jobs in parallel.

## D: stronger fault detection

The green table permits honest qualified misses. This track asks the harder
question: how many real concurrency faults can each flavor actually diagnose,
especially on the end-to-end model rather than only on a micro-test?

```mermaid
flowchart TB
  D0["D0 DONE: exact identities, mutation<br/>cardinality and outcome schema"]:::done
  D1["D1: measure Qwen Sampled detection<br/>under untuned standard-v1 defaults"]:::todo
  D2["D2: improve Qwen SC/RR/Sampled/Inline<br/>detection without user site knowledge"]:::todo
  D3["D3: complete atomic address, order,<br/>scope, RMW, CAS and fence micro-matrix"]:::todo
  D4["D4: derive admitted e2e instances of<br/>useful micro-matrix fault families"]:::todo
  D5["D5: execute clean/fault controls with<br/>diagnostic attribution and miss rates"]:::todo
  D6["D6: publish per-flavor sensitivity,<br/>overhead and memory Pareto view"]:::todo
  DG{"REAL-WORLD DETECTION<br/>STRENGTH CHARACTERIZED"}:::target

  D0 --> D1 --> D2
  D0 --> D3 --> D4
  D2 --> D5
  D4 --> D5 --> D6 --> DG

  classDef done fill:#70ad47,stroke:#274e13,stroke-width:2px,color:#000;
  classDef todo fill:#a5a5a5,stroke:#404040,stroke-width:2px,color:#000;
  classDef target fill:#ed7d31,stroke:#843c0c,stroke-width:3px,color:#000;
```

`D1` is important because the retained Qwen Sampled detection rate belongs to
the explicit stride-256 fault experiment, while ordinary clean use now selects
`standard-v1` stride 16,384 automatically. Characterize the default before
deciding whether a different automatic policy is warranted. An experimental
multi-trial sweep may expose controls internally; the ordinary user contract
must not require tuning them.

Flavor contracts remain distinct: SuperCollider is a redundant-access
perturbation/check engine; Record/Replay observes its named bounded snapshot;
Sampled makes a statistical claim; Inline should attribute deterministic
causal or shadow-state diagnostics. Do not make a weak flavor look strong by
silently changing its memory or trial budget.

## H: adversarial-input hardening

ConSan is used on programs already suspected of synchronization mistakes.
Ill-formed final images, inconsistent metadata, failed publication, or a bad
injected mutation must fail with a bounded, typed outcome rather than hang,
crash, corrupt output silently, or reset the GPU.

```mermaid
flowchart TB
  H0["H0 DONE: malformed-input taxonomy,<br/>bounded transform controls and regressions"]:::done
  H1["H1: independent final-image inventory<br/>and generated-byte reconciliation"]:::todo
  H2["H2: corrupt identity, owner, capacity,<br/>ordering, transaction and CAS state"]:::todo
  H3["H3: fuzz transformed control flow,<br/>caves, relays and spill restoration"]:::todo
  H4["H4: host parser/renderer checks against<br/>the exact final generated image"]:::todo
  H5["H5: bounded multi-wave/multi-launch live<br/>clean and corrupt-state qualification"]:::todo
  HG{"ADVERSARIAL INPUTS<br/>FAIL SAFELY"}:::target

  H0 --> H1 --> H4
  H0 --> H2 --> H4
  H0 --> H3 --> H4
  H4 --> H5 --> HG

  classDef done fill:#70ad47,stroke:#274e13,stroke-width:2px,color:#000;
  classDef todo fill:#a5a5a5,stroke:#404040,stroke-width:2px,color:#000;
  classDef target fill:#ed7d31,stroke:#843c0c,stroke-width:3px,color:#000;
```

This preserves the useful intent of the former Inline final-image DAG without
making it a retroactive prerequisite of the already-qualified workload cells.
See [MALFORMED_INPUT.md](MALFORMED_INPUT.md) for the normative taxonomy.

## B: additional real workloads and vocabulary

Do not resume the historical reject list or the abandoned 410-site
SuperCollider relay objective in the abstract. Broaden coverage only when a
concrete admitted real workload demonstrates the need.

```mermaid
flowchart TB
  B0["B0 DONE: current workspace north-star<br/>workloads have runnable contracts"]:::done
  B1["B1: admit one new e2e workload with<br/>assets, oracle and baseline command"]:::todo
  B2["B2: inventory exact unsupported sites,<br/>event vocabulary and resource blockers"]:::todo
  B3["B3: implement the smallest shared fix<br/>with focused host regression"]:::todo
  B4["B4: qualify four clean/fault/overhead<br/>cells and rerun Qwen sentinel"]:::todo
  BG{"NEW REAL-WORKLOAD<br/>ROW GREEN"}:::target

  B0 --> B1 --> B2 --> B3 --> B4 --> BG

  classDef done fill:#70ad47,stroke:#274e13,stroke-width:2px,color:#000;
  classDef todo fill:#a5a5a5,stroke:#404040,stroke-width:2px,color:#000;
  classDef target fill:#ed7d31,stroke:#843c0c,stroke-width:3px,color:#000;
```

An isolated kernel may be the shortest reproducer for `B3`, but it does not
become a status-table row unless it is independently valuable. New vocabulary
must retain typed exclusions for unsupported forms rather than weakening the
denominator or passing through silently.

## P: another gfx architecture

```mermaid
flowchart TB
  P0{"gfx1201 final-tip release<br/>certificate available"}:::target
  P1["P1: acquire target GPU and record<br/>driver, ISA and toolchain identity"]:::deferred
  P2["P2: generate target inventory and<br/>classify ISA/ABI applicability gaps"]:::todo
  P3["P3: port patching, spill, report and<br/>fault-injection mechanisms"]:::todo
  P4["P4: run clean/fault/overhead matrix<br/>with target-specific health gates"]:::todo
  P5["P5: publish STATUS_<ARCH>.md and<br/>cross-architecture differences"]:::todo
  PG{"NEW gfx TARGET<br/>QUALIFIED"}:::target

  P0 -. requires target hardware .-> P1
  P1 --> P2 --> P3 --> P4 --> P5 --> PG

  classDef todo fill:#a5a5a5,stroke:#404040,stroke-width:2px,color:#000;
  classDef target fill:#ed7d31,stroke:#843c0c,stroke-width:3px,color:#000;
  classDef deferred fill:#9e7cc1,stroke:#351c75,stroke-width:2px,color:#000;
```

Do not infer architectural portability from gfx1201 host tests. Reuse the
validation scripts and experiment schema, but re-establish applicability,
final-byte mutation identities, clean correctness, diagnostics, performance,
memory, and device health on the target hardware.

## Maintenance rules

- Keep `STATUS_RDNA4.md` concise and evidence-backed; it is a result ledger,
  not a worklog.
- Update affected Mermaid node colors in the same commit as implementation or
  evidence changes.
- Use a new empty artifact directory after source, hook, manifest, or failed
  preparation changes; never mix provenance.
- Do not use coverage-limiting or workload-specific tuning for ordinary
  acceptance. Expert fault-campaign controls must be disclosed separately.
- Commit after each bounded node or coherent sibling group.
- Do not run more than four GPU jobs in parallel.
- Do not work on another architecture without suitable hardware.
