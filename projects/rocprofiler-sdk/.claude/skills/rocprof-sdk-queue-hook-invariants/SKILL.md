---
name: rocprof-sdk-queue-hook-invariants
description: "Use when touching the HSA queue interception path in rocprofiler-sdk — adding or moving a subsystem that injects AQL packets, editing the activation predicates or client ids under hsa/queue_hooks/, or merging a feature across the queue-callback removal. Catches the silent failures where a subsystem's hook is never reached or a whole feature is skipped, with no build error."
---

# Queue Hook Invariants — rocprofiler-sdk

Queue interception has no central registry. Each subsystem exposes its own `write_hook`, `signal_completion_hook`, and `is_any_active`, and `WriteInterceptor` asks two named predicates in [hsa/queue_hooks/activation.hpp](../../../source/lib/rocprofiler-sdk/hsa/queue_hooks/activation.hpp) whether to do any work at all. Every failure mode here is silent: the code compiles, the tests pass, and packets go through uninstrumented.

## Invariant 1 — Predicate Completeness

Every subsystem that installs a write hook must appear in **every** predicate that guards the path to it.

| Missing from | Consequence |
|--------------|-------------|
| `any_consumer_active()` | `WriteInterceptor` forwards the submission untouched. No signal is allocated, no packet is rewritten, and the subsystem's hook is never called. A single-subsystem session collects nothing |
| `should_batch_packets()` | A multi-packet submission is processed as one batch, so only the first dispatch is instrumented. Any subsystem that injects per-dispatch packets must veto batching |

The predicates must be named once and called by both production and tests. A test that restates the expression inline passes while `queue.cpp` disagrees.

## Invariant 2 — Client Ids Are Pairwise Distinct

Completion routes each `inst_pkt_t` entry back to its producer by the tag in [client_ids.hpp](../../../source/lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp). A duplicated id hands one subsystem's packet to another's post-dispatch handler. Only distinctness matters, so assert it directly rather than pinning values.

## Invariant 3 — A Feature Merged Onto This Path Must Admit Itself

The gate runs before interception, so a feature doing its work inside `WriteInterceptor` is skipped entirely unless it is part of the gate condition — as kernel replay is, via `has_active_replay_contexts()` beside `any_consumer_active()`. A conflict resolution that merges cleanly and compiles can still drop the feature completely.

The neighboring trap is a signature narrowing back during conflict resolution: `Queue::sync()` returning `bool` (drain succeeded) reverting to `void` changes behavior at every call site with no compile error.

## Adding or Moving a Subsystem — Checklist

- [ ] `queue_hooks.{hpp,cpp}` in the subsystem directory exposing `write_hook`, `signal_completion_hook`, `is_any_active`
- [ ] A new client id, and a test asserting all ids are pairwise distinct
- [ ] Added to `any_consumer_active()`, and to `should_batch_packets()` if it injects per-dispatch packets
- [ ] A test that asserts both predicates **from a state where the subsystem is active** — the assertion is vacuous otherwise
- [ ] `write_hook` verified inert with no active context, and verified to tag its packets with its own id

## Diagnosing "the refactor predates the subsystem"

A refactor branched before a subsystem landed will not mention it, and nothing about that reads as an error. Establish it in one step instead of guessing:

```bash
git merge-base --is-ancestor <refactor-branch> origin/develop
git ls-tree --name-only <refactor-branch>:<path-to-subsystem>
```

An empty listing means the refactor never saw the subsystem, so every predicate it introduced is missing an entry.

## Merge Tactic

Before merging across this path, extract your feature out of the shared hot spot. Moving a large block out of `WriteInterceptor` into a subsystem-owned `queue_hooks.cpp` turns an unresolvable conflict in the file's most-edited function into a few mechanical hunks, and makes the block testable on its own.

## Open Question

Completion keys off the active-context list, so a dispatch already in flight when a context stops loses its data. Routing completion by the producer id already present in `inst_pkt_t` would close that window. Undecided — do not silently "fix" it as part of unrelated work.
