# Emulation review checklist

Select only sections relevant to the change. This is a risk checklist, not a
requirement to produce one comment per item.

## Correctness and lifecycle

- Bounds, overflow, alignment, signedness, truncation, wraparound, and units.
- Null/empty/partial inputs and malformed external data.
- Error propagation, stable error codes, rollback, retries, and idempotence.
- Ownership and cleanup for allocations, mappings, file descriptors, handles,
  processes, threads, sockets, temporary files, and persistent state.
- Initialization and teardown ordering, including failure halfway through.
- Compatibility of public C APIs, CLI output/exit status, JSON/schema state,
  environment variables, and on-disk formats.

## Concurrency and process behavior

- Shared fields have one documented synchronization discipline.
- Lock order is consistent; no lock is held across blocking I/O, callbacks,
  RPC, process waits, or potentially re-entrant code without justification.
- Atomics use an ordering that establishes the required happens-before edge.
- Worker/poll/RPC/simulation threads cannot outlive referenced state.
- Shutdown, cancellation, timeout, signal, fork, exec, and crash recovery paths
  wake all waiters and leave recoverable state.
- Multi-process identifiers, runtime directories, ports, FIFOs, and sockets are
  isolated; tests remain reliable under parallel execution.

## rocjitsu architecture and domain

- KFD ioctls and process behavior stay in the simulated driver; interposition
  remains a translation boundary rather than a second implementation.
- Vendored or kernel UAPI types define kernel ABI; convenience library types do
  not accidentally become wire formats.
- Queue/doorbell/event lifecycle, dispatch ordering, VM mappings, cache
  coherence, and signal semantics match the intended contract.
- Simulation work does not block engine progress or violate deterministic event
  ordering.
- Hot paths avoid exceptions, unnecessary allocation, logging, and contention.
- DBT preserves control flow, registers, wait/order semantics, code-object
  metadata, and behavior across every supported source/target ISA pair.

## Generated code

- Generated ISA and DBT files were not manually edited.
- Generator or semantic source changed in the same patch.
- Multi-ISA regeneration covers every affected family and pair.
- Shared semantics remain shared instead of copied into one ISA.
- Generated diff is deterministic, comprehensive, formatted, and free of
  unrelated version/tool drift.
- amdisa tests and representative decode/execute/translation tests cover the
  source change.

## Mirage

- Crate ownership and feature gating remain coherent; optional features compile
  independently where supported.
- FFI loading validates symbols and lifetimes and does not let Rust references
  outlive loaded libraries or C-owned storage.
- Profile/session/execution transitions remain atomic and crash recoverable.
- XDG paths, permissions, cleanup, process groups, PTYs, FIFOs, and signals work
  for concurrent sessions.
- Container command construction preserves argument boundaries, mount and
  environment behavior, rank/topology wiring, and useful errors.
- HTTP/WebSocket changes preserve authentication assumptions, input validation,
  cancellation, backpressure, and client-visible compatibility.
- UI state remains synchronized with server state; asynchronous work handles
  stale responses, unmount, reconnect, and errors.
- Integration coverage uses the owning suite in `emulation/mirage/tests/`, such
  as lifecycle, daemon, container, or matrix E2E tests.

## Style and documentation

- Re-read `emulation/rocjitsu/docs/style.md` for every rocjitsu review.
- Check type choice and section order, naming, namespaces, C++20/STL usage,
  header form, `auto`, logging, exceptions, formatting, and comments.
- Public behavior and non-obvious invariants are documented where maintainers
  and users will look. Avoid duplicated or narrational documentation.

## Test selection

- Add a regression test that fails before the fix and exercises the public or
  owning layer, plus focused unit tests for boundary-heavy logic.
- Include negative, boundary, cleanup, and concurrency cases when those paths
  changed.
- Run rocjitsu unit tests for engine changes and Mirage lifecycle/integration
  tests when users reach the behavior through Mirage.
- For dashboard changes, run lint, production build/type-check, unit tests, and
  user-flow E2E tests as warranted.
