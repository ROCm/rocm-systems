# Shared Utilities

The `util` library provides low-level building blocks shared across
rocjitsu and simdojo. The `util` CMake target supplies public headers and a
static library containing the statistics registry and JSONL exporter. Most
other utilities and the owner-local statistics accumulators are header-only.

## Headers

### Bit Manipulation

| File | Description |
|------|-------------|
| `bit.h` | Bit masks, insertion, extraction, counting, alignment. Provides `mask()`, `insert_bits()`, `bits()`, `is_power_of_2()`, `align_up()`. |
| `bitfield.h` | Typed bitfield access over integral or array storage. `Bitfield<N>` with templated `get<>()` / `set()` for arbitrary bit ranges. |

### Data Types and Conversions

| File | Description |
|------|-------------|
| `data_types.h` | Floating-point format conversions: FP16, BF16, FP8 (E4M3), BF8 (E5M2), FP4, FP6. Scalar and vectorized block converters with IEEE-754 rounding modes (truncation, round-to-nearest-even, stochastic). |
| `convert.h` | Type-erasing conversion between opaque C API handles and internal object pointers. |

### SIMD

| File | Description |
|------|-------------|
| `simd.h` | Vectorized operations via `<experimental/simd>` with scalar fallback. Float conversions, integer arithmetic (`mul_hi`, `popcount`, `clz`), rounding, and transcendental approximations (`rcp`, `rsq`, `log`, `exp`). |
| `simd_test_hooks.h` | Test-only seam for forcing scalar execution to compare against SIMD paths. |

### Memory and Concurrency

| File | Description |
|------|-------------|
| `arena_alloc.h` | Fixed-size block pool allocator with free-list. `ArenaAlloc<BlockSize, NumBlocks, BlockAlign>` provides O(1) alloc/dealloc with global-allocator fallback. |
| `spinlock.h` | TTAS spinlock optimized for sub-microsecond critical sections. Uses C++20 `atomic::wait()` with ThreadSanitizer annotations. |
| `conditional_wake_spinlock.h` | Parking lock for many independent, rarely contended critical sections where release cost dominates. A caller-supplied host policy provides timed waits and wakes, keeping native/interposed API selection outside util. Linux uses a store release; a missed wake waits up to the current timeout (capped at 2 ms) plus scheduling delay. Sustained contention may be slower than `Spinlock`. The lock must outlive every in-flight `unlock()` call; other hosts use `Spinlock`. |
| `intrusive_list.h` | Bidirectional intrusive linked list with optional parent-pointer tracking. `IntrusiveList<T>` and `IListNode<T>` with O(1) insert/erase. |
| `unique_handle.h` | Move-only RAII ownership for POSIX file descriptors, Win32 handles, and custom handle traits. |
| `intrusive_shared_ptr.h` | Strong-only shared ownership with object-provided ownership hooks, plus atomic `RefCounted<T>` and counter-isolated `RefCountedCacheAligned<T>` storage helpers. `RefCountedNonAtomic<T>` supports non-atomic ownership when all ownership operations, destruction, and count inspection stay on one thread or are externally synchronized across all handles sharing the object. |

### Diagnostics

| File | Description |
|------|-------------|
| `log.h` | Compile-time and runtime configurable logging with named groups (VM, CP). Thread-safe, supports lazy evaluation via lambda forms. Use `Logger::print<GroupId>()` for group-filtered output. |
| `result.h` | No-throw `Result` and `FailureOr<T>` return types for expected failures. |
| `diagnostic.h` | Non-owning diagnostic sinks and stream-style error construction for result-returning APIs. |
| `except.h` | Exception hierarchy: `Exception` (base), `InvalidInst`, `UnimplementedInst`, `ConfigError`. All derive from `std::exception`. |

### Statistics

`statistics.h` supplies owner-local accumulators and a reusable statistics
registry in the standalone `stats` namespace. The namespace describes the
package; its headers and compiled implementation belong to the `util` target.
Neither API depends on the simulator, ISA, configuration loader, or plugin
system. The accumulators have no clock or output dependency; callers must
synchronize their shared updates and snapshots.

| Type | Description |
|------|-------------|
| `stats::Counter` | Monotonic `uint64_t` counter. Addition and merging saturate on overflow and retain an explicit overflow flag. |
| `stats::Gauge` | Current `int64_t` value. An unset gauge is distinct from an observed zero. It has no implicit merge policy. |
| `stats::Summary` | Mergeable count, sum, minimum, maximum, mean, and population variance. Nonfinite input and arithmetic overflow set an explicit invalid flag. |
| `stats::Histogram` | Fixed, finite, strictly increasing upper bounds with inclusive endpoints and an unbounded final bucket. Each bucket accumulates samples in its own interval over the recorder's lifetime; merging requires identical bounds. |

Construct histogram layouts with `stats::Histogram::create()` before recording.
Updates and successful merges allocate no storage. Snapshot values own their
data, including histogram bounds and counts, and can outlive the accumulator.
An empty summary has count zero and placeholder numeric fields. An invalid
summary retains a saturated count of finite samples, but its numeric aggregates
must be treated as unavailable. Neither snapshots nor accumulator merges
establish a concurrency barrier.

For histogram bounds `{0, 4, 8}`, snapshot counts `{2, 2, 2, 1}` describe seven
samples: two in each bounded interval and one above `8` in the unbounded bucket.
The counts are not prefix sums across buckets. `HistogramSnapshot::overflowed`
means at least one bucket counter exceeded `UINT64_MAX` and saturated; a sample
in the unbounded bucket does not indicate overflow. The summary separately
reports count overflow through `SummarySnapshot::count_overflowed`.

The compiled registry uses the same header:

| Type | Description |
|------|-------------|
| `stats::RegisteredCounter`, `stats::RegisteredGauge` | Registered atomic values with owned descriptors and stable shared handles. |
| `stats::RegisteredSummary`, `stats::RegisteredHistogram` | Registered distributions protected by a metric-local mutex. Recording does not allocate. |
| `stats::Scope` | Named collection of metrics with copied attributes, metric selection, and descriptor-conflict detection. |
| `stats::Session` | Registry and output owner. Creates an exclusive JSONL file and exports a final snapshot after producers stop. |
| `stats::Config` | Metric selection, output directory and format, schema identifier, and default measurement domain. |

Generic scopes use caller-supplied keys and display names. Applications can add
JSON metadata to scope, run, and session-end records through metadata writers;
the writers append comma-prefixed JSON members and must not reenter the
registry. `stats::json_string()` escapes string values. Scope writers own
their captures for the scope lifetime. The session adds no background thread;
call `finish(complete)` only after producers are quiescent. Registration conflicts
make the exported snapshot incomplete even when the caller passes `true`.

Rocjitsu's adapter in `rocjitsu/stats/statistics.h` supplies GPU identities,
component/dispatch selection, the dispatch scope limit, build provenance, reset
metadata, and VM completion policy. Those concepts are not part of the generic
utility API.

The default `util.statistics.v1` JSONL output is accepted by
`scripts/rocjitsu-stats.py` for SQLite and CSV conversion. The rocjitsu adapter
adds its own metadata under the also-supported `rocjitsu.statistics.v1` schema.

### Miscellaneous

| File | Description |
|------|-------------|
| `dynamic_loader.h` | Runtime symbol resolution via `dlsym` (Linux) / `GetProcAddress` (Windows). `lookup_symbol<T>()` returns a typed function pointer. |
| `meta_programming.h` | Type traits and concepts: `IsIntegral`, `IsUnsigned`, `IsArithmetic`, `IsVoid`. Also provides `GetOption` for template option extraction and `always_false_v<T>` for deferred static assertions. |

## Usage

All headers are under `lib/util/include/util/` and can be included as:

```cpp
#include "util/log.h"
#include "util/bit.h"
```

Consumers link the `util` target in CMake to obtain its include path and
compiled statistics implementation. It does not link rocjitsu core libraries.
