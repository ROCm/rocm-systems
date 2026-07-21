# ConSan future work

[STATUS_RDNA4.md](STATUS_RDNA4.md) records the gfx1201 north-star matrix. The
2026-07-16 final-tip pass found contrary evidence in five cells; focused
repairs and one-tip clean, fault, and overhead campaigns resolved them. The
gfx1201 release certificate is complete. The remaining tracks separately
improve actual fault-detection strength, adversarial robustness, workload
breadth, or hardware portability.

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
  S["RDNA4 clean ledger and Qwen fault<br/>requalified at one tip"]:::done
  R["R DONE: repair contrary evidence and<br/>issue release certificate"]:::done
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

The four RDNA4 tracks are technically independent. `R` converted heterogeneous
retained cell evidence into a single third-party-reproducible release
certificate. Work on `D`, `H`, or `B` should rerun the highest-value affected
e2e sentinel immediately.

## R: final-tip release certificate

This completed track compresses the former broad acceptance DAG into evidence
that can be reproduced by another developer.

```mermaid
flowchart TB
  R0["R0 DONE: validation runner, audit view,<br/>workspace contract and current cell evidence"]:::done
  R1["R1 DONE: freeze executable 640e575da2,<br/>rebuilt hook c45aa0fe, device/assets"]:::done
  R2["R2 DONE: run all 11 north-star clean rows;<br/>55/55 baseline/profile rows pass"]:::done
  R2A["R2A DONE: fix reserved ISA fields and<br/>clause relocation; TP2 Inline clean"]:::done
  R2B["R2B DONE: tree Atomic-OR Inline<br/>complete and clean"]:::done
  R3["R3 DONE: Qwen exact drop accepted:<br/>SC 1/1, Sampled 16/32, Inline 1/1"]:::done
  R4["R4 DONE: all 14 admitted exact<br/>fault policies pass"]:::done
  R5["R5 DONE: 66/66 paired no-fault<br/>overhead result rows pass"]:::done
  R6["R6 DONE: audit hashes, zero omissions,<br/>no-knob commands and generated summaries"]:::done
  R7["R7 DONE: update STATUS_RDNA4 from<br/>one-tip script outputs"]:::done
  RG{"DONE: FINAL-TIP gfx1201<br/>RELEASE CERTIFICATE"}:::done

  R0 --> R1 --> R2
  R2 --> R2A --> R3
  R2 --> R2B --> R3
  R3 --> R4 --> R5 --> R6 --> R7 --> RG

  classDef done fill:#70ad47,stroke:#274e13,stroke-width:2px,color:#000;
```

Acceptance requires a clean committed tree and a hook rebuilt from exactly that
tree. Run no more than four GPU jobs in parallel; serialize fault rows capable
of destabilizing the device. Do not use the repository-wide CTest inventory as
a concurrent GPU certificate: it mixes unrelated fixtures that are not
mutually isolated. Run the 2,069-test host binary directly, and use this
validation runner for live workload rows. Every result retains pre/post GPU
health. Ordinary profiles must have no workload-specific coverage or sampling
settings.

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
