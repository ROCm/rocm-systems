---
name: rocjitsu-kernel-parity
description: Compare rocjitsu's simulated Linux KFD/AMDGPU behavior with a Linux source checkout and derive focused tests. Use when reviewing or implementing KFD ioctls, DRM/AMDGPU interposition, queue, doorbell, event, memory mapping, process, topology, or error semantics, or when asked whether rocjitsu matches kernel behavior.
compatibility: Requires read access to a Linux source checkout, normally ~/linux; supplemental documents may exist under ~/reference.
metadata:
  author: ROCm
  version: "1.0"
---

# Rocjitsu kernel parity

Establish whether rocjitsu reproduces the externally observable Linux
AMDGPU/KFD behavior relevant to a change. The kernel is a behavioral oracle,
not an architecture template: rocjitsu may implement behavior differently if
the observable contract remains correct and the project layering is preserved.

## Hard rules

1. Follow `emulation/AGENTS.md` and the current
   `emulation/rocjitsu/docs/style.md`.
2. Default to read-only analysis. Never modify the Linux checkout. Modify
   rocjitsu or tests only when the user explicitly asks for implementation.
3. Record the exact rocjitsu commit and Linux commit or describe why either is
   unavailable. Do not compare against an assumed kernel version.
4. Trace behavior from public entry point through validation, state mutation,
   synchronization, and cleanup. Matching a struct declaration is not proof of
   matching behavior.
5. Public shader programming guides, ISA manuals, and architecture documents
   may be stored in `~/reference/public/shader-programming-guides`; users can
   obtain public copies from
   [AMD GPU architecture programming documentation](https://gpuopen.com/amd-gpu-architecture-programming-documentation/).
   Keep confidential PDFs separately in `~/reference/confidential`.
6. Confidential references are supplemental only. Never quote, name, link,
   copy, cite, summarize, upload, or expose their contents through paths,
   metadata, screenshots, logs, prompts, generated artifacts, internal
   terminology, issues, pull requests, reviews, tests, or chat. Findings must be
   independently justified by repository code, publicly identifiable Linux
   source, public programming documentation, or reproducible tests. Treat
   uncertain publication status as confidential.
7. Separate verified mismatch, intentional emulation difference, version skew,
   and unknown. Do not label an uncertainty as a kernel-parity bug.
8. Never push or otherwise write to GitHub without explicit user approval
  immediately before that specific write. Analysis, implementation, commit,
  review, and pull-request preparation requests do not imply publication
  approval; approval for one write does not authorize another.

## 1. Pin scope and sources

Identify the exact operation and observable behavior under question: ioctl,
`mmap`, event wait, queue lifecycle, process teardown, topology, memory policy,
doorbell, exception, or error code. Record relevant inputs, process/thread
state, expected outputs, and the hardware-independent behavior being modeled.

Locate the Linux tree, normally `~/linux`, and record:

- `git rev-parse HEAD`, nearest release/tag if useful, and whether the tree is
  clean;
- AMDGPU/KFD UAPI declarations and implementation files reached by the
  operation;
- configuration or version conditions that select alternate behavior.

If the requested kernel revision is not present, ask for it or clearly bound
the conclusion to the available commit. Do not fetch or checkout over local
Linux work without permission.

## 2. Build a behavior trace

Use [the parity checklist](references/kernel-parity-checklist.md). Trace both
sides independently before comparing them:

1. entry point and ABI layout;
2. validation order and first returned error;
3. lookup, ownership, permissions, and process/device context;
4. locks, references, lifetime, and concurrency;
5. state changes and side effects;
6. copy-to/from-user behavior and partial failure;
7. cleanup, rollback, close, process exit, and device loss;
8. version, capability, and architecture gates.

In rocjitsu, inspect the interposer, simulated driver, VM/model ownership, RPC
or daemon path, and tests. Verify the behavior is implemented in the correct
layer rather than duplicated between interposer and simulation.

## 3. Compare observable behavior

Create a compact matrix with rows for meaningful cases and columns for Linux,
rocjitsu, evidence, and status. Include success plus relevant invalid, missing,
duplicate, boundary, racing, and teardown cases. Compare:

- accepted inputs and validation precedence;
- return values, `errno`, output fields, and side effects;
- object identity, reference ownership, and cross-process visibility;
- ordering, wakeup, blocking, timeout, and cancellation behavior;
- memory alignment, offset, size, mapping, coherence, and lifetime rules;
- behavior after partial setup, close, process exit, reset, or failure.

Do not require internal algorithm or lock-for-lock parity. Require observable
parity and sufficient synchronization for rocjitsu's own threading model.

## 4. Prove or bound each mismatch

For each candidate mismatch:

- verify the complete Linux and rocjitsu paths, including error and cleanup;
- check existing tests, comments, history, UAPI version gates, and deliberate
  emulator limitations;
- construct the smallest practical rocjitsu regression test or non-mutating
  repro that distinguishes the behaviors;
- run focused tests when possible, then the relevant broader CTest; exercise
  the behavior through Mirage when Mirage owns setup or environment wiring.

If hardware is needed, describe the missing experiment and expected observable
result; do not imply it ran. Private material may increase confidence but may
not appear as evidence.

## 5. Report

Start with one of: `MATCH`, `MISMATCH`, `INTENTIONAL DIFFERENCE`,
`VERSION-DEPENDENT`, or `INCONCLUSIVE`.

For every mismatch provide:

- rocjitsu `file:line` and public Linux `file:line`, with both commits;
- triggering inputs and state;
- Linux behavior, rocjitsu behavior, and user-visible impact;
- why the difference is not merely an implementation detail;
- a minimal fix in the correct rocjitsu layer;
- a regression test, including Mirage integration when relevant;
- confidence and any unverified condition.

End with the behavior matrix, tests run, unavailable checks, and remaining
version/hardware risk. If confidential references were consulted, do not
mention that fact in the report.
