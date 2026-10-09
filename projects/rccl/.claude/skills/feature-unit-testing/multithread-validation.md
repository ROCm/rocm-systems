# Multithread Validation of a Net Plugin

Module of the *feature-unit-testing* skill. Read it when a transport test suite
needs to cover concurrent use of a net plugin, or when deciding which
concurrency is worth testing at all.

## Pick the concurrency the product actually performs

A plugin-level test replaces a production thread with a host worker thread, so
first establish which production thread makes each call:

- the proxy progress thread of a communicator drives the data path
  (`isend` / `irecv` / `test`);
- the proxy service thread of the same communicator performs connection setup
  and memory registration (`listen` / `connect` / `accept` / `regMr`);
- the plugin context is created per communicator, and communicators derived
  from a parent share both the context and the proxy state.

So one process with several independent communicators runs several plugin
contexts, each driven by its own proxy threads. That — N independent contexts
and connections, used concurrently inside one process — is the shape worth
testing, and it is what frameworks produce when a single process owns several
devices or several communication groups.

Two shapes are *not* worth testing:

- **Concurrent calls on one communicator.** One progress thread owns a
  communicator, so the product never does this. The send and receive comms hold
  shared FIFO, request, and work-request state with no per-comm serialization; a
  test would be a data race whose green result carries no information.
- **Concurrent plugin `init` / `finalize`.** The core serializes these under its
  own plugin mutex, so the race is unreachable from the product. Verify this in
  the core before writing such a test — if it holds, the test only exercises the
  test harness.

Confirm both claims in the current source rather than assuming; they are
properties of the core, and they decide whether a test is meaningful.

## Worker option

`--net_ib_nthreads=N` selects how many host workers each MPI rank uses in the
tests that support it. Parse and strip a custom flag before GTest sees argv,
since `InitGoogleTest` does not know about it.

Requirements worth enforcing in the parser, because a silently ignored value
turns a multithread configuration into a single-threaded run:

- reject anything that is not a positive integer, and reject conflicting
  repeated values;
- clamp to a maximum so a typo cannot exhaust the machine;
- reduce the value across all ranks and fail before the suite starts unless
  every rank resolved the same one.

## Choose which tests get a threaded variant

Pick by the state concurrency would actually stress, not by how easy the body is
to convert. In descending order of value:

- **Registration paths.** The MR cache and protection domain are per physical
  device behind one mutex, so registration storms, refcount churn on a single
  shared address, and fused-device cycling (where one registration fans out
  across both members) are the only way to race insert, lookup and eviction.
- **Fault injection whose state is per communicator.** One worker breaking its
  own connection while the others keep transferring is an assertion no serial
  test can make. Concurrent link failures are also the only way to load a
  process-global recovery thread.
- **Per-communicator scheduler state**, such as WRR tokens and cursor: every
  worker arms and audits its own ledger, so N schedulers advance at once while
  sharing nothing.
- **Data paths under pressure** — size ladders, multi-QP splits, request-pool
  churn, GPU buffers. There is no software race to find in per-comm state; the
  value is real queue contention plus proof that completion routing never
  crosses communicators.

Give every worker a distinct payload seed. A transfer delivered on the wrong
connection then fails verification instead of passing silently, which is the one
cheap defence against the whole model being wrong.

Leave alone anything that asserts on process-global state: device and merged-NIC
tables, counters snapshotted process-wide, and timing claims about a shared
background thread. Resource-leak checks still work, but only on the main thread
around the entire run — an in-loop snapshot sees other workers' live resources
and reports a leak that is not there.

Spread workers across devices when the suite has several NICs. Pinning every
worker to device 0 leaves the other devices' protection domains and caches
untouched.

## Harness structure

- The main thread creates every context and connection, exchanges handles over
  MPI, and tears everything down after joining. Workers never call MPI: a
  worker-side collective deadlocks the peer rank when a sibling fails.
- Workers return a result struct instead of using GTest assertions, and the main
  thread reports those results after joining. GTest fatal assertions are
  thread-unsafe and a `return` inside a worker cannot fail the test.
- Per-thread MPI tag namespaces keep the main-thread handshakes separate. Assert
  at compile time that the worst-case tag fits the range every MPI
  implementation must provide.
- Synchronize a rank-wide failure flag between phases, so both ranks abandon a
  connection handshake together instead of one waiting forever.

## Prove that the workers overlapped

A threaded test that a scheduler serialized still passes while covering nothing.
Make overlap a test assertion:

- release the workers from a common gate, then hold them at a second gate so no
  worker can finish its body before the others have entered theirs — the
  observed maximum of concurrently active workers becomes deterministic instead
  of timing-dependent;
- fail the test when the distinct thread count is below the requested worker
  count, or when the observed overlap is less than two;
- cancel the gates when thread creation fails, so already-started workers do not
  wait forever.

Keep the single-worker path byte-for-byte as it was, and gate the threaded path
behind the option. The default run then stays a regression check for the
original behavior.

## What such tests are worth

They exercise the state the plugin shares between contexts on one device —
protection domain references, the memory-registration cache, device tables,
asynchronous port recovery — under real concurrency, plus the per-connection
data path in parallel. They say nothing about collective semantics or
performance; those belong to suite-level and framework-level tests.
