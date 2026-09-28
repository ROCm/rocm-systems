# Torch Trace Collector

## System context

Torch tracing records the PyTorch calls around GPU work so analysis can
attribute kernel counters to operators. The workload emits one ROCTX range
per call. Each range describes that call, its available arguments, and its
correlation metadata; analysis reconstructs relationships between ranges.

The native collector runs inside the workload process. Python structural
wrappers record module and API calls with user source locations. A global
PyTorch RecordFunction callback records operators on every executing thread,
including autograd workers. When the native collector is unavailable,
TorchDispatchMode records operators on the Python thread where it is active.

This design covers the producer, its runtime loading, and its marker contract.
The analysis consumer owns call-tree construction and kernel attribution.
Triton wrappers use the same marker contract for their launch entry points.

## Requirements

- Emit one range for each observed call, preserving normal ROCTX nesting on
  the executing thread without repeating an ancestor path in each label.
- Keep Python structural wrappers active with both native and fallback
  operator tracing.
- Observe all RecordFunction scopes, including backward functions and user
  scopes, with available sequence numbers and PyTorch thread identifiers.
- Capture bounded operator arguments, including tensor shapes, dtypes, and
  schema argument names when available.
- Publish the Python autograd launcher's operating-system thread identifier
  to worker callbacks so analysis can attach worker intervals to that call.
- Build and package one C++17 shared library without PyTorch or Python build
  dependencies. Profiling loads that library without compiling it.
- Use native tracing for PyTorch 2.13 and 2.14. Warn and use TorchDispatchMode
  for other minors or when discovery, loading, or installation fails.
- Contain failures at the native callback and plain-C boundaries, and balance
  every successful range push and launcher publication.

## Components and data flow

| Component | Responsibility | Output or dependency |
| --- | --- | --- |
| Python structural wrappers | Record module and API calls with user locations; publish the launcher thread around selected autograd entry points | Python ROCTX ranges and a native launcher publication |
| Python collector loader | Check the supported PyTorch minor, locate the generic artifact, promote the workload's libtorch, and bind the plain-C interface | One installed native callback or fallback |
| RecordFunction collector | Observe calls on executing threads and combine correlation fields with captured arguments | One ROCTX range per callback |
| Private ABI shim | Declare the small PyTorch surface and layouts needed by the collector | Runtime-resolved Torch/c10 and AOTI operations |
| Profile output handling | Preserve marker and counter CSVs for each collection pass | Raw marker Function text, thread ids, timestamps, and counters |
| Analysis consumer | Parse metadata, reconstruct interval nesting, and correlate GPU dispatches | Operator call trees and statistics |

Python and native producers emit their own ranges. They do not share an
application-maintained marker stack. The collector does not save forward
stacks or merge ancestor names into worker markers.

## Correlation decisions

### RecordFunction and Python wrappers

RecordFunction provides operator names and metadata on autograd workers,
where Python dispatch instrumentation is inactive. Structural wrappers add
source locations and high-level names that RecordFunction does not provide.
Native tracing replaces TorchDispatchMode for operator events while preserving
the structural wrappers.

### Launcher thread propagation

Before a wrapped autograd launch, Python publishes its native thread id using
PyTorch ThreadLocalDebugInfo. Autograd carries that debug info to the worker
task. Worker callbacks record the inherited id in the launcher field, and the
Python wrapper restores the preceding publication when the call ends.

The launcher id is an operating-system thread id, matching the marker CSV's
Thread_Id. It is distinct from the PyTorch thread ids used with sequence
numbers. Analysis can use the launcher id and interval containment to attach
worker ranges to the launcher's call. These identifiers remain separate in
the marker contract.

### Flat marker contract

The one-level marker contains the call name, source location, sequence number,
PyTorch thread id, forward thread id, launcher thread id, scope, arguments,
and backend. Unknown values use the existing unavailable sentinel. The
backend remains a trailing marker token, and profile preserves the Function
cell unchanged.

The wire syntax and storage contract are detailed in the
[low-level design](lld-torch-trace-collector.md). Consumers of the earlier
stacked labels need the one-level parser; the collector does not translate
between those formats.

## Build and runtime compatibility

The shipped artifact is named torch_trace_collector.so. Its name is independent
of the workload's PyTorch and Python versions. Production compilation uses
C++17, local shim declarations, and rocprofiler-sdk-roctx. PyTorch headers,
PyTorch libraries, and Python development headers are not build dependencies.

At runtime, the Python loader imports the workload's PyTorch, accepts the
2.13 and 2.14 minors, and promotes that installation's libtorch_cpu.so into
the global symbol scope. It then loads the collector locally with ctypes.CDLL
and validates revision 2 of its four-function plain-C interface. The loader
retains the library handles because PyTorch keeps the callback installed for
the process lifetime.

The private PyTorch C++ layout is not a stable upstream ABI. Supporting a
minor means its recorded layouts have been validated, not that every custom
build of that minor is guaranteed compatible. The accepted policy uses the
minor-version gate and real-header tests; exact wheel identities and ELF
build ids are not runtime acceptance criteria. AOTI version information used
to interpret dtype differences is separate from that acceptance policy.

## Validation and extension points

Native validation links real libtorch and compares the shim's layouts and
behavior with upstream headers. ROCPROFCOMPUTE_TORCH_ROOT selects that test
dependency; it is not needed to build the production collector. Loader tests
cover generic-artifact discovery, accepted and rejected minors, the plain-C
revision, runtime symbol promotion, and fallback. Marker and integration tests
cover arguments, scopes, balanced ranges, and launcher propagation.

Adding another supported PyTorch minor requires validating the private layouts
and symbol behavior before extending the loader's supported set. Changes to
the marker fields or encoding must be coordinated with the analysis consumer
and both Python and native producers. The generic artifact and profile CSV
layout remain independent of those future extensions.
