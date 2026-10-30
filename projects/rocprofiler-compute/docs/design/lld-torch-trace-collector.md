# LLD: Torch trace collector

## Scope and source ownership

This document describes the producer defined by the
[high-level design](hld-torch-trace-collector.md). It emits flat ROCTX markers
and leaves call-tree construction to analysis.

| Source | Responsibility |
| --- | --- |
| [torch.py](../../src/utils/inject_roctx/_backends/torch.py) | Structural Python wrappers and TorchDispatchMode fallback |
| [torch_trace_collector.py](../../src/utils/inject_roctx/_backends/torch_trace_collector.py) | Generic-artifact discovery, runtime loading, C-interface binding, and launcher publication |
| [core.py](../../src/utils/inject_roctx/core.py) | Python ROCTX runtime emission and source-location resolution |
| [marker_format.py](../../src/ml_api_tracing/marker_format.py) | Shared Python marker encoding, composition, parsing, and legacy stacked-wire detection |
| [torch_trace_collector.cpp](../../src/lib/torch_trace_collector/torch_trace_collector.cpp) | Process-wide callback installation, range lifetime, and launcher debug info |
| [argument_capture.cpp](../../src/lib/torch_trace_collector/argument_capture.cpp) | Bounded input rendering and schema-name lookup |
| [wire_format.h](../../src/lib/torch_trace_collector/wire_format.h) | Native marker serialization |
| [torch_abi.h](../../src/lib/torch_trace_collector/torch_abi/torch_abi.h) | Measured private PyTorch layouts used by the local declarations |
| [utils_profile.py](../../src/utils/utils_profile.py) | Copy marker and counter CSVs into the workload directory |

## Installation and callback lifetime

The loader imports PyTorch and reduces its version to major.minor. Only 2.13
and 2.14 select native tracing. It locates torch_trace_collector.so using
native_tool_finder, with installed artifacts searched before source build
directories.

The loader reopens the workload's own libtorch_cpu.so with global symbol
visibility. The collector is then loaded locally through ctypes.CDLL, so its
unresolved Torch/c10 and AOTI symbols resolve from the workload. Both library
handles are retained for the process lifetime. No Python extension ABI crosses
this boundary.

After checking the C-interface revision, the loader binds all entry points
and installs one global RecordFunction callback. Installation is serialized
and idempotent. The callback requests inputs and observes every RecordFunction
scope. The Python backend activates TorchDispatchMode only if native loading
or installation fails; structural Python wrappers remain active in either
case.

For each RecordFunction start, the callback borrows the current name,
correlation fields, and inputs for the duration of that callback. It captures
the arguments, formats one marker, and pushes one ROCTX range. Its observer
context tells the matching end callback whether that push succeeded. End pops
only a successfully pushed range. The observer context does not retain tensor
or RecordFunction pointers.

Exceptions are contained before returning through RecordFunction or plain-C
entry points. A failed callback does not issue an unmatched range pop.

## Launcher propagation

Selected Python autograd wrappers publish the current native operating-system
thread id immediately before calling the wrapped entry point. The native
collector stores that id in a LauncherTidInfo object under its dedicated
ThreadLocalDebugInfo key. PyTorch carries the shared debug info into autograd
worker tasks; a worker callback reads the inherited value for its marker.

Publications may nest. The native code tracks successful pushes per thread,
and the Python wrapper pops only after its own push succeeded. The pop restores
the previous publication even when the wrapped workload raises. A worker keeps
the shared debug info for the task lifetime; the marker never contains a
pointer to a Python frame or object.

The launcher id matches the CSV Thread_Id namespace. RecordFunction's tid and
ftid fields use PyTorch's own thread identifiers and must not replace it.

## Argument capture

The callback requests RecordFunction inputs. The capture module reads the
borrowed input-array view through the shim's recorded offset and uses the
operator name and overload, when available, to look up schema argument names.
The collector updates cached argument names when PyTorch registers or removes
a schema. It reads schemas under PyTorch's registration lock and keeps old
name data until process exit so other threads can finish reading it safely.
If names are unavailable, arguments are recorded without them.

| Input | Marker representation |
| --- | --- |
| Defined tensor | Dtype followed by dimensions separated by x inside brackets |
| Undefined tensor | None |
| TensorList | A bracketed list of rendered tensors, limited to eight entries |
| Other IValue | Its type tag rather than its full contents |
| No inputs or unavailable capture | n/a |
| Individual value capture failure | ? |

Tensor metadata is read through runtime AOTI operations. The module does not
copy tensor contents or retain their ownership. Common dtype names come from
AOTI dtype identifiers; version-dependent fallback dtype names use the AOTI
version independently of the loader's supported-minor check.

TensorLists borrow their existing element array during synchronous capture.
Traversal stops after eight entries or when the text buffer truncates, without
copying the list or retaining element ownership.

Capture renders at most 32 top-level arguments and limits the unencoded text
to 512 bytes of payload. Truncated text may add the four-character closing
suffix, and percent encoding can expand the wire field further. Python
wrappers apply the corresponding character-count limit and delimiter encoding
to their available tensor arguments.

## Marker wire contract

The existing one-level v1 syntax is an encoded name, a colon, and its location,
followed by pipe-delimited fields in the order below. The v1 label describes
the wire contract; it is not an additional serialized token. The native C API
revision is versioned separately.

| Field | Native RecordFunction value | Python wrapper value |
| --- | --- | --- |
| Name | Current RecordFunction name | Wrapped API or module name |
| Location | n/a | User file:line when available, otherwise n/a |
| seqNr | Nonnegative sequence number, otherwise n/a | n/a |
| tid | Current PyTorch thread id | n/a |
| ftid | RecordFunction forward thread id, including zero when absent | n/a |
| ltid | Inherited launcher OS thread id, otherwise n/a | n/a |
| scope | RecordFunction scope name, otherwise n/a | n/a |
| args | Encoded, bounded argument text, otherwise n/a | Encoded available arguments, otherwise n/a |
| Backend | Raw trailing torch token | Raw trailing torch or triton token |

The metadata field names are followed by an equals sign. The backend is a
bare final token. Name encoding replaces percent and slash with %25 and %2F.
Argument encoding replaces percent, pipe, semicolon, carriage return, and
newline with %25, %7C, %3B, %0D, and %0A. Source locations retain the existing
unencoded representation.

Each label contains one call. Parent-child relationships come from ROCTX
timestamps and executing thread ids, not from slash-separated ancestor paths.
Python wrappers and native callbacks must preserve the same field names and
encoding for downstream parsing.

## CSV storage and analysis boundary

Profile copies each pass's marker and counter CSVs into the workload directory
with the existing ml_api_trace prefix and marker_api_trace/counter_collection
suffixes, preserving compression. The Function cell stays unchanged: the
backend, argument text, and correlation fields are not split into columns at
collection time.

A compatible analysis consumer parses the flat fields, nests marker intervals
per OS thread, and uses ltid plus timestamp containment to attach worker trees
to their launcher. Across replay passes, ltid is not a stable identity because
the OS assigns new thread ids. The stitch key is the Function cell with
ltid= removed: name, location, seqNr, tid, ftid, scope, args, and backend.
function_ordinal is the 0-based occurrence of that stitch key in the pass,
in marker order, assigned after join. Cross-pass collapse matches on stitch
key and function_ordinal. A regex match for `:#\d+@` in the first
`|`-separated Function token identifies legacy stacked wire and makes current
analyze exit during parse.

## Analyze path

`--list-*-operators` / `--*-operator`:

1. Pair each `ml_api_trace_*_marker_api_trace.csv` with its sibling
   `_counter_collection.csv`.
2. Rename on-disk `Correlation_Id` to `Correlation_ID`, then full-outer-join
   unique dispatches to markers on `Correlation_ID` and `GUID`.
3. Consolidate matching operator calls across passes on the stitch key plus
   `function_ordinal`. Collapsed GPU time and kernel lists come from pass 0.
4. Parse Function into name, file/line, `Backend`, `launcher_thread_id`
   from `ltid=`, and args. `tid=` and `ftid=` stay on the Function cell
   for stitch; they are not analyze columns.
5. Nest marker intervals per `Thread_Id`. Adjacent ranges (`A.end == B.start`)
   are siblings.
6. Attach unlocated worker trees to the launcher using `ltid` and interval
   containment.

Integrity failures abort analyze before the tree: overlapping non-adjacent
intervals on the same `Thread_Id`, a marker missing from the nested forest,
unequal kernel-name and timestamp lists on a marker, or operator or kernel
disagreement across passes. Coverage issues warn during join and attach, and
the forest continues: unmatched dispatches, attach failures, and unlocated
roots in `KNOWN_ML_API_BACKENDS` with no launcher thread. Plain analyze
without those flags does not join and does not report those conditions.

## Plain-C interface

The collector exports exactly four functions at interface revision 2.

| Entry point | Contract |
| --- | --- |
| torch_trace_collector_abi_revision | Return the interface revision |
| torch_trace_collector_install | Install the process-wide callback; return zero on success |
| torch_trace_collector_push_launcher_tid | Publish a uint64 launcher OS thread id; return zero on success |
| torch_trace_collector_pop_launcher_tid | Restore this thread's preceding successful publication; return zero on success |

Each non-query operation reports failure with a nonzero status. Python binds
the complete interface before installation and verifies the revision before
using it. The callback remains installed until process exit. This interface
does not expose the former native user-scope stack, snapshot store, uninstall,
or statistics operations.

## Build and private ABI

Production builds produce one C++17 torch_trace_collector.so using local shim
declarations and rocprofiler-sdk-roctx. They require neither a PyTorch
installation nor Python development headers. PyTorch symbols intentionally
remain unresolved until runtime. The installed artifact belongs in the
rocprofiler-compute library directory under lib or lib64, without a Torch
minor or Python-SOABI suffix.

The local shim covers RecordFunction and callback layouts, input views,
IValue payloads, ListImpl vector begin/end positions and the libstdc++ pointer
representation, operator names and handles, Dispatcher registration listeners
and their registration handle, and ThreadLocalDebugInfo. These
are private PyTorch interfaces. The accepted compatibility boundary is the
validated 2.13/2.14 layout set, not an upstream ABI guarantee for every build
of those minors. Extending this set requires real-header and runtime behavior
validation. No exact wheel identity or paired ELF build-ID gate is used.

## Tests

Live profile covers torch, triton, and combined ML API traces, plus operator
coverage against the native collector. Unit tests cover inject_roctx wraps,
Function parse, join, nest, and operator list/filter.

`ROCPROFCOMPUTE_TORCH_ROOT` selects real headers and libraries for optional
native layout and behavior tests. It is a test dependency only. Those tests
compare shim assumptions with real PyTorch types and exercise the C boundary,
argument rendering, all scopes, balanced callbacks, and launcher propagation.
Loader and wrapper tests verify discovery, supported minors, revision
rejection, runtime promotion, fallback, and successful-push/pop pairing.

`tests/integration/test_profile_torch_trace.py` also checks tensor arguments,
scopes, launcher propagation, and copied marker and counter CSVs.
