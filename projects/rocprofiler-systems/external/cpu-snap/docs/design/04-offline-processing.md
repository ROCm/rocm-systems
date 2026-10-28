# 04 - Offline processing and output

This is the in-depth design document for the cold side of cpu-snap: the passes
that run AFTER `sampler::stop()` and turn captured `raw_sample` records into a
trace. It documents the four components of the offline render top-down, with the
WHY for every class and decision: `symbolizer`, `sample_postprocessor` (plus
`processed_sample` / `processed_frame` and the free function
`coalesce_frames_by_name`), `frame_filters`, and `slice_emitter` (plus
`emitted_slice`). It also states precisely why the work is split the way it is,
and where the backend-neutral output boundary sits.

It deliberately does NOT re-derive the layering, the capture path, or the wire
format. Read those first:

- [./00-overview.md](./00-overview.md) - the three layers, the end-to-end data
  flow, and where this cold side fits.
- [./01-core.md](./01-core.md) - the `src/core/` data plane, including `proc_maps`
  (the offline `/proc/self/maps` reader these passes lean on) and `reader`
  (which turns a streamed byte stream back into `raw_sample`s).
- [./02-capture-and-backends.md](./02-capture-and-backends.md) - the `sampler`, the
  two backends, and how a `raw_sample` is produced. The cold side here starts
  where that document ends.
- [./03-streaming-and-wire.md](./03-streaming-and-wire.md) - the live encode and
  the wire format; the streaming path feeds this render side via
  `reader::resolve_stack`.
- [./06-design-decisions.md](./06-design-decisions.md) - the decision log.
  Section 10 there is the one-paragraph summary that this document expands.

Reference docs (user-facing ground truth, linked rather than restated):

- [../components.md](../components.md) - per-component API and dependents for all
  29 components.
- [../wire-format.md](../wire-format.md) - the v1 byte layout.
- [../usage.md](../usage.md) - task-oriented how-to (symbolization, filtering,
  output modes).
- [../extending.md](../extending.md) - the output-sink extension point, which is
  exactly the `emitted_slice` boundary documented here.
- [../architecture.md](../architecture.md) - prose orientation.
- [../../planning/architecture-sampler-abstraction.md](../../planning/architecture-sampler-abstraction.md)
  - the locked plan behind the capture seam; the offline side is the unchanged
  downstream it protects.

Audience: a developer who must understand and maintain the whole project.

---

## 1. Where this fits, and why there is a cold side at all

Capture (doc 02) produces a per-tid map of `raw_sample` POD records. Each record
is `{tid, timestamp_ns, ip, frames}` with `frames` innermost-first
(matching the kernel callchain order). That is all the hot path is allowed to do:
fill a fixed POD with no allocation, possibly in an async-signal context. It does
NOT symbolize, does NOT filter, and does NOT shape a timeline, because every one
of those operations allocates, can shell out to `addr2line`, or builds a map -
none of which belong on a path that runs while the target is executing.

So everything expensive is deferred to a separate set of passes that run after
`stop()`. This is the single organizing idea of the whole library (doc 06,
section 1): a tiny fixed-cost hot path, and a large amount of cold work pushed
behind it. The offline render is that cold work for callstacks.

```
   capture (doc 02) -> streaming bytes (doc 03)
   sampler -> sink_fn -> reader.feed(bytes)
        |  resolve_stack(leaf) / core::sample_reconstructor
        |  -> per-tid vector<raw_sample>
        v
   +----------------------------------------------------------------------+
   |                  THIS DOCUMENT: offline render                       |
   |                                                                      |
   |   symbolizer        IP -> "function [binary]" (+ inline + file:line) |
   |       ^  resolver / inline_resolver callbacks                        |
   |       |                                                              |
   |   sample_postprocessor   per-frame resolve + predicate + sink        |
   |       |   -> processed_sample / processed_frame                      |
   |       |                                                              |
   |   slice_emitter          timeline / nested-slice shaping             |
   |       |   -> emitted_slice   (BACKEND-NEUTRAL OUTPUT BOUNDARY)       |
   |       |                                                              |
   |   frame_filters     library-frame predicate shared by both consumers |
   +----------------------------------------------------------------------+
        |
        v   emitted_slice / processed_sample
   output backend  (the example Perfetto writer -> .pftrace;
                    or any future pprof / folded-stacks / rocpd sink)
```

Every capture streams to a sink. The byte stream is replayed through
`core::reader`, whose `resolve_stack(leaf_id)` returns the innermost-first IP
chain, which a consumer repacks into `raw_sample`s (`core::sample_reconstructor`
does this and returns the per-tid map). From the moment a `raw_sample` exists,
the render side is identical regardless of how the sample was captured or where
the bytes were stored. That convergence is the payoff of the
backend-neutral downstream principle (doc 00, section 6): the render code is
written once against the captured 4-tuple, not against perf, signal, or the wire.

### 1.1 Why four components, not one

The cold side is split into four single-purpose pieces because they answer four
genuinely different questions, run at different points, and have different
outputs:

- `symbolizer` answers "what is the human name (and source location, and inline
  chain) for this raw IP?" It is a pure lookup service. It owns the only call out
  to `addr2line` and the only use of `proc_maps` for address translation.
- `sample_postprocessor` answers "which frames and which whole samples survive my
  predicates, and what does each survivor look like once symbolized?" Its output
  is `processed_sample`/`processed_frame` - per-sample, per-frame records aimed at
  a consumer that wants filtered, named frames (a folded-stacks writer, a custom
  aggregator).
- `slice_emitter` answers "what does the timeline look like - which frames are
  open over which time spans?" Its output is `emitted_slice` - timeline spans
  aimed at a trace viewer (Perfetto).
- `frame_filters` is the one piece of logic the postprocessor and the emitter
  genuinely share - dropping shared-library frames - factored out so it is not
  duplicated. It is the audit's one real de-duplication (doc 06, section 12).

The postprocessor and the slice emitter are NOT merged into one pass even though
both walk the same per-tid `raw_sample` map, because their jobs and outputs are
disjoint: one does predicate/symbol/sink delivery on individual frames, the other
does open/close timeline shaping across consecutive samples. A trace writer wants
slices; a flamegraph-collapsed writer wants filtered named frames. Keeping them
separate means each output format reaches for exactly the pass it needs, and a
new format does not inherit machinery it does not want. They share only what is
genuinely common (the library filter), and that sharing is the entire reason
`frame_filters` exists as its own component rather than living inside either
consumer.

---

## 2. `symbolizer` - batched IP to "function [binary]"

`src/cpu_snap/processing/symbols/symbolizer.hpp` (header-only,
namespace `cpu_snap::sampling::utils`). It is the lookup service the other two
passes call through: given a raw runtime instruction pointer, it returns a
formatted function name, optionally the full inline-call chain, and optionally a
`file:line` source location.

### 2.1 The problem it solves, and why batching is the design

Turning an IP into a name requires two steps: (1) find which binary owns the IP
and translate the runtime address into that binary's file offset, and (2) ask a
symbolizer (`addr2line`) for the function at that offset. Step (1) is
`proc_maps`; step (2) is a subprocess.

The subprocess is the cost that dictates the design. Invoking `addr2line` once
per IP would spawn one process per address - hopeless for a real trace with
thousands of unique IPs. So the symbolizer is built around `preload(ips)`, which:

1. groups all IPs by their owning binary path (via `proc_maps::find` and
   `ip_to_file_offset`), then
2. runs `addr2line` at most once per binary, passing every offset for that binary
   on a single command line.

This collapses thousands of `popen` calls to one per distinct loaded module. The
batching is the whole point of the class; it is why `symbolizer` exists as a
stateful object with a cache rather than a free `resolve(ip)` function. (Doc 06,
section 10: "per-IP `addr2line` would be hopeless for a large trace.")

The typical caller flow, from `examples/single_thread_callstack_demo.cpp`:
collect the unique set of IPs across every sample (the leaf `ip` plus every
`frames[i]`), call `preload(unique_ips)` once, then resolve each IP through the
cache during slice emission.

### 2.2 The cache shape and what each lookup returns

The cache is keyed by raw IP and stores, per IP, the full inline-call chain as a
`std::vector<std::string>` (`m_cache`), plus a separate `m_file_line` map of
IP -> "file:line". Storing the whole chain (not just one name) is what lets the
single cache serve both the outer-name-only contract and the inline-expansion
contract without re-running `addr2line`.

The chain vector is innermost-first: index 0 is the function the IP physically
lives in (often the only leaf the optimizer left), each subsequent entry is an
enclosing inlined frame, and the last entry is the outer non-inlined function.
This orientation matters because the public accessors slice it differently:

- `resolve(ip)` -> `string_view` of the OUTERMOST frame (`chain.back()`). This is
  the legacy "one name per IP" contract. Side-effect-free; returns an empty view
  if the IP was never preloaded.
- `resolve_inline(ip)` -> the chain. When `symbolizer_config::inline_frames` is
  true it returns the full vector; when false it returns a single-element vector
  holding just `chain.back()`, so a caller that always uses `resolve_inline` gets
  `resolve`-equivalent output when the feature is off and the full chain when it
  is on, without changing call sites.
- `file_line(ip)` -> `string_view` of the captured `file:line`, or empty.
- `operator()(ip)` -> an owned `std::string` of the outermost name. This is the
  `std::function`-compatible single-IP entry point designed to plug straight into
  `sample_postprocessor_config::resolver` / `slice_emitter_config::resolver`. On a
  cache miss it falls back to a single-IP `preload({ip})` (one `popen`), and in
  the worst case returns `"0x<hex>"`. The note on the method is load-bearing:
  prefer `preload` for bulk work; this fallback path costs one subprocess per miss.

The cache also makes `preload` idempotent: an already-cached IP is skipped, so
re-preloading after a `rescan_maps()` only resolves the genuinely new addresses.

### 2.3 `symbolizer_config` - the two knobs

`symbolizer_config` has exactly two fields, and the minimalism is deliberate:

- `include_binary_name` (default true) - append `" [binary]"` to each resolved
  name, e.g. `"__nanosleep [libc.so.6]"`. When false the output is just the
  function. The same toggle governs the unresolved case: an IP that maps to a
  binary but cannot be symbolized renders as `"0x<hex> [binary]"` or `"0x<hex>"`;
  an IP that maps to no binary at all is always `"0x<hex>"` (there is no binary
  to attach). `format_with_binary(name, base)` is exposed publicly so the format
  policy can be unit-tested independently of `addr2line` and `proc_maps`.
- `inline_frames` (default false) - opt into inline-chain expansion in
  `resolve_inline`. The documented cost is roughly 2-3x more frames per sample in
  the trace; the symbolizer's preload time is unchanged because `addr2line -ifCe`
  is invoked the same way either way - only the parse and the cache shape differ.
  Default false preserves pre-feature behaviour for callers that do not opt in.

### 2.4 The `addr2line` invocation, parsing, and the sentinel trick

`build_addr2line_cmd` constructs `addr2line -ifCe "<path>" 2>/dev/null` followed
by the per-IP offsets. The flags carry rationale:

- `-i` surfaces the full inline-call chain per IP (innermost first). This is what
  populates the chain vector even when `inline_frames` is false (the data is
  always captured; the config only governs how much of it `resolve_inline`
  hands back).
- `-f` prints the function name, `-C` demangles, `-e` names the executable.
- There is deliberately NO `-p`: without it each entry is TWO lines (function,
  then `file:line`), so an inline chain reads as N consecutive line-pairs, which
  the parser walks pair by pair.

The hard part of batching multiple IPs through one `addr2line` invocation is
knowing where one IP's (possibly multi-frame) inline chain ends and the next IP's
begins. The solution is a SENTINEL address `0xffffffffffffffff` injected between
consecutive IPs on the command line. `addr2line` emits `"??\n??:0"` for it, so
`read_one_chain` reads function/file pairs until it hits a `"??"` function
(`is_sentinel_function`), consumes that separator pair without recording it, and
stops. This is the mechanism that makes one subprocess per binary reliable
instead of guessing chain boundaries by counting.

Supporting parse details, each with a reason:

- `clean(fn)` strips `" [clone ...]"` tags and `"(anonymous namespace)::"`
  noise but deliberately KEEPS the parameter list - callers want full signatures,
  so only genuine noise is removed.
- `normalize_file_line` maps the `"??:0"` / `"??:?"` / empty sentinels to an
  empty string, so `file_line(ip)` returns empty rather than a junk location.
  Only the FIRST non-sentinel pair in a chain owns the IP's `file:line` (the
  physical leaf's source location).
- On `popen` failure the batch falls back to a `"<basename>+0x<ip>"` name per IP
  (`fallback_hex_for`) so a missing or unreadable `addr2line` degrades to a
  binary-relative hex address rather than losing the frame.
- An IP whose chain comes back empty still gets a single hex-name entry, so every
  preloaded IP always has at least one cache entry.

The `popen`/`pclose` calls are flagged `NOLINT(cert-env33-c, concurrency-mt-unsafe)`
because the batched per-binary `addr2line` genuinely requires a subprocess; this
is an offline pass, not the hot path, so the cost and the mt-unsafety are
acceptable.

### 2.5 `rescan_maps` and lifetime

`rescan_maps()` re-reads `/proc/self/maps` (via `proc_maps::reload`) while
PRESERVING the existing cache, so previously resolved IPs keep returning the same
string. This supports a long offline session where modules were loaded after the
symbolizer was first constructed. The symbolizer owns its `proc_maps` instance
outright, so unlike `frame_filters` (section 4) it has no external-lifetime
requirement on the maps object.

---

## 3. `sample_postprocessor` - resolve, filter, deliver

`src/cpu_snap/processing/symbols/sample_postprocessor.hpp` (header-only,
namespace `cpu_snap::sampling`). This is the per-frame resolve + predicate +
sink pass. Its reason for existing is one sentence (the file comment): it splits
expensive work - symbol resolution, predicate filtering, user callbacks - OUT of
the hot recording path. Run it after `stop()`.

### 3.1 Output types: `processed_frame` and `processed_sample`

- `processed_frame` is `{uint64 ip, string_view resolved}`. The `resolved` view
  references the postprocessor's internal symbol cache and is stable for the life
  of the instance (or until `clear_cache()`). It may be empty when no resolver
  was configured or the resolver could not symbolize the IP.
- `processed_sample` is `{tid, timestamp_ns, ip, span<const processed_frame>
  frames}`. The `frames` span points into the postprocessor's per-call scratch
  buffer and is valid ONLY inside the sink call. The warning is explicit and
  load-bearing: copy out anything you need to retain past the sink return.

This view-into-owned-storage design is what keeps the pass allocation-light: the
symbol strings live once in a cache keyed by IP, the per-sample frame list reuses
one scratch vector, and `processed_frame`/`processed_sample` just borrow. Nothing
is copied per frame except the `{ip, view}` pair.

### 3.2 `sample_postprocessor_config` - the knobs

- `resolver` - `std::function<std::string(uint64 ip)>`. Called at most once per
  unique IP; the result is cached. Null resolver means every `resolved` view is
  empty. This is where a `symbolizer` is plugged in (its `operator()` matches the
  signature).
- `frame_predicate` - `std::function<bool(uint64 ip, string_view name)>`. Return
  false to drop a frame. The `name` is the cached resolver output (may be empty).
- `drop_library_frames` (default false) - the convenience toggle that lazily
  builds a `proc_maps` and the `make_main_exe_only_filter` predicate from
  `frame_filters` (section 4) on first use. It composes with `frame_predicate` by
  AND: a frame survives iff the library filter accepts it AND the user predicate
  (when set) accepts it. Setting this is exactly equivalent to chaining that
  filter manually - the field exists purely so the common "main-exe only" case is
  one bool instead of three lines of wiring.

The callbacks are `std::function` - and that is fine here. This is the cold
path; a thunk once per frame in an offline pass costs nothing measurable, and the
flexibility (any callable, including a `symbolizer` or a lambda over one) is worth
it. This is the deliberate inverse of the hot path's de-thunking (doc 06,
sections 2 and 10): flexibility is recovered downstream precisely because it is
free there.

### 3.3 The per-sample pipeline

`process(span<raw_sample>, sink)` iterates records; `process_all(by_tid, sink)`
iterates a whole per-tid map (the shape `core::sample_reconstructor::samples()` returns). Both forward
to `process_one`, which per `raw_sample`:

1. `ensure_library_filter()` lazily constructs the `proc_maps` + filter on first
   use when `drop_library_frames` is set (so an instance that never enables it
   never pays for a maps read).
2. For each of the frame IPs: resolve (cached), then apply the library
   filter, then the user `frame_predicate`. A frame dropped by either bumps
   `m_dropped_frames` and is skipped; survivors are pushed into the scratch vector
   as `processed_frame{ip, name}`.
3. Build the `processed_sample` over the scratch span.
4. Bump `m_delivered` and call `sink(out)`.

`resolve_cached` is where symbolization is memoized: on a cache miss it calls the
resolver inside a `try`/`catch(...)` that swallows any throw and leaves the name
empty, then stores it. A resolver failure therefore degrades one IP to an
anonymous frame rather than aborting the pass - the same best-effort philosophy
the wire side uses (doc 03, section 6). The `sink` parameter is taken by lvalue
reference, not forwarded, because it is invoked many times per call; the
`NOLINT(cppcoreguidelines-missing-std-forward)` documents that the non-forwarding
is intentional.

The instance is move-only (copy deleted) because it owns a cache, scratch, and a
`unique_ptr<proc_maps>`. It is single-threaded by contract.

### 3.4 Telemetry

`delivered_samples`, `dropped_frames`, and `unique_symbols`
are cumulative counters a host reads after the pass to reconcile what survived.
`clear_cache()` drops the memoized resolutions for the case where the binary
mapping changed between runs.

### 3.5 `coalesce_frames_by_name` - the slice-count fix

This free function (alongside the postprocessor, same header) is small but solves
a concrete trace-quality bug, and the rationale in its comment is worth keeping in
mind whenever you touch slice emission.

The kernel frame-pointer walker emits slightly DIFFERENT IPs inside the same
function across consecutive samples - different basic blocks of the one function
land on different addresses. The nested slice emitter (section 5) diffs
consecutive stacks by IP to decide which frames stayed open. If the same function
shows a different leaf IP on every sample, the diff sees the stack change every
time, so it closes and reopens that frame on every sample, exploding the slice
count into one open/close cycle per sample even though the function never changed.

`coalesce_frames_by_name(samples, resolver)` fixes this by rewriting every frame
IP in place to a single CANONICAL IP per resolved function name: the first IP
seen for a given name becomes the canonical address, and every later IP that
resolves to the same name is rewritten to it. After this pass, "same function"
means "same IP", so the emitter's IP diff correctly keeps the frame open. IPs that
resolve to an empty name are left as-is (each is its own canonical, since there is
no name to group on); resolver throws are swallowed and treated as anonymous.

It is applied AFTER recording and BEFORE slice emission. Its workload-time cost is
zero - it is an offline mutation of already-recorded data, and the sampler always
emits raw IPs unchanged. The resolver is called at most once per unique IP
(internal `ip_to_canonical` / `name_to_canonical` maps). It is an alternative to
the slice emitter's `inline_resolver` path (which diffs on names directly); a
caller using IP-based nested emission wants this pass, a caller using the
inline-name path does not need it.

---

## 4. `frame_filters` - the shared library-frame predicate

`src/cpu_snap/processing/symbols/frame_filters.hpp` (header-only,
namespace `cpu_snap::sampling::utils`). Two builder functions return
`std::function<bool(uint64 ip, string_view)>` predicates. They are the genuine
common denominator between the postprocessor and the slice emitter, and that is
the whole reason they live in their own component instead of inside one of the
consumers (doc 06, sections 10 and 12; the audit's one real de-duplication).

- `make_main_exe_only_filter(maps)` - keep a frame iff its IP belongs to the main
  executable mapping, OR falls outside every known mapping (anonymous / JIT
  pages, which are kept on the assumption they are the host's own code), and drop
  everything that lands in a shared library (libc, libstdc++, ld-linux, vDSO).
  The main-exe path is resolved once at construction via
  `core::self_exe_path()` (falling back to `maps.main_exe()`), then each
  call does one linear `proc_maps::find` scan plus a path/`is_main_exe` check.
  This is the predicate the `drop_library_frames` toggles wire up.
- `make_library_exclude_filter(maps, excluded)` - drop a frame whose owning
  module path contains ANY of a caller-supplied substring list (e.g.
  `{"libc.so", "libstdc++.so"}`); a frame outside any known mapping is kept. This
  is the more surgical "exclude these specific libraries" variant.

The critical lifetime contract, stated in the file comment: both builders capture
a REFERENCE to the caller-owned `proc_maps`, which MUST outlive the returned
predicate. The builders snapshot only the main-exe path by value; the mapping
lookups go through the live reference. This is why both consumers own the
`proc_maps` instance (via `unique_ptr`) for at least as long as the filter they
build from it - see the postprocessor's `m_library_maps` and the emitter's
`library_frame_filter::m_maps`.

The duplication this resolves was real: before extraction, both the postprocessor
and the slice emitter had their own copy of "build a main-exe-only predicate over
a proc_maps". Now both call the same builder. The postprocessor uses it as a
frame-level drop inside `process_one`; the emitter uses it to compact each
`raw_sample`'s frame array before shaping (section 5.5). Same predicate, two
application points.

---

## 5. `slice_emitter` - timeline shaping and the output boundary

`src/cpu_snap/processing/timeline/slice_emitter.hpp` plus
`src/cpu_snap/processing/timeline/slice_emitter.cpp` (namespace `cpu_snap::processing::timeline`, component
`cpu_snap::processing::timeline`). This is the timeline-shaping pass. It converts per-tid
`raw_sample` vectors into a stream of `emitted_slice` records and hands each to a
sink callback. It does NO symbolization, filtering, or coalescing itself - those
live in the symbolizer, the postprocessor, and `coalesce_frames_by_name`; this
module is purely the timeline shaper, and it links only `cpu_snap::core` (no
`cpu_snap::processing::symbols`, no libdw). The header/`.cpp` split exists because the
templated entry points (`process` / `emit_nested`) must inline at the call site
against the caller's sink type, while the non-template helpers (name
resolution, library-frame filtering) are out-of-line.

### 5.1 `emitted_slice` - the backend-neutral output boundary

```cpp
struct emitted_slice
{
    pid_t            tid;       // sampled thread
    std::string_view name;      // slice label (borrowed)
    std::uint64_t    start_ts;  // stream clock ns (CLOCK_MONOTONIC by default)
    std::uint64_t    end_ts;    // stream clock ns (CLOCK_MONOTONIC by default)
    std::uint32_t    depth;     // nesting depth, 0 = outermost
};
```

This struct is the single most important interface in the offline render, because
it is the seam at which the library stops and output formats begin. Everything
upstream of it is cpu-snap; everything downstream is a format-specific sink. The
Perfetto writer is one such sink, but it is demo glue
(`examples/common/perfetto_proto_writer.hpp`), not an installed library target -
which is the proof that the boundary holds: a pprof writer, a folded-stacks
writer, or a rocpd writer would each be a new sink over `emitted_slice` with zero
changes to the library. This is one of the two documented extension points
([../extending.md](../extending.md), section 2).

`depth` lets a sink that rebuilds begin/end ordering from completed spans break
same-timestamp ties (open shallow-first, close deep-first), since timestamps
alone do not encode nesting when a whole sub-stack shares one sample's interval.

`name` is BORROWED from emitter-owned per-call storage and valid only inside the
sink call. A sink that retains it must copy it out - which the example Perfetto
writer does (it copies `name` into an owned string when buffering each span).

The timestamps are nanoseconds of the stream clock (`CLOCK_MONOTONIC` by default), matching the perf path, the
streaming `sample_record`, and the signal backend's wall-time stamps (doc 06,
section 4) - one clock end to end, so a sink never has to reconcile clocks.

### 5.2 `slice_emitter_config` - knobs and why each exists

- `merge_samples` (default true) - when true, a frame that stays on the stack
  across consecutive samples is one slice spanning all of them (the flame-graph
  view). When false, nothing is carried over: every sample emits its own full
  stack as nested slices over `[its timestamp, the next sample's)`, so the
  timeline shows exactly what each sample captured (one slice per frame per
  sample, so the output is large).
- `sample_period_ns` (default `k_default_sample_period_ns` = 1 ms, i.e. a 1 kHz
  fallback) - how far past the final sample to close still-open frames (and how
  long the last sample's region lasts when `merge_samples` is false). It exists
  because the last sample has no successor to bound its end timestamp, so the
  emitter has to assume one sampling period of duration.
- `resolver` - `std::function<std::string(uint64)>`, IP -> name. If empty the
  emitter installs `default_hex_name` (the `"0x<addr>"` fallback) in its
  constructor, so the emitter always has a resolver and never has to null-check.
- `inline_resolver` - optional IP -> innermost-first vector of formatted frame
  names. When set, each sampled IP is expanded into its inline
  chain and the emitter shapes one slice per chain entry, diffing on NAMES rather
  than IPs (section 5.4). This is the alternative to `coalesce_frames_by_name`:
  diffing on names inherently coalesces same-named frames, and it also surfaces
  inlined sub-frames as their own slices.
- `symbol_identity` - optional IP -> symbol-identity key
  (`std::optional<uint64_t>`), used only by the non-inline `emit_nested` IP diff
  (section 5.3). When the common-prefix walk diverges at depth `cp`, the emitter
  checks whether the two diverging IPs still share a `symbol_identity` key before
  falling back to a raw-IP mismatch; if they do, the frame stays open instead of
  spuriously closing and reopening. This is what makes a function called from
  multiple call sites collapse to one slice instead of splitting once per call
  site. Unset = always fall back to raw-IP equality (pre-existing behaviour).
- `library_frame_filter` - optional `std::function<bool(uint64_t,
  string_view)>` per-frame keep/drop predicate, applied to compact each sample's
  frame array before shaping (section 5.5). Unset (default) = no filtering.
  `timeline` does not construct or own a module source itself: the caller builds
  this predicate (typically via `cpu_snap::processing::symbols::make_main_exe_only_filter`,
  fed from a live `proc_maps` or a `core::module_map`) and passes it in
  ready-made, which is what keeps `timeline` link-independent of `symbols` and
  `core::proc_maps`.

### 5.3 `process` and the nested IP diff (the core algorithm)

`process(samples, sink)` walks every tid, copies the tid's samples into a local
vector, sorts that vector by `timestamp_ns` (samples can arrive out of order
across a drain batch), optionally applies the library filter, then dispatches to
`emit_nested`. It returns the total slice count.

`emit_nested` (the IP path) is the heart of the timeline shaper. It maintains a
stack of `open_frame{ip, start_ts}` representing the frames currently open, and
for each sample:

1. Build `new_stack` by reversing the sample's innermost-first `frames` into
   OUTERMOST-first order (root at index 0), because slices nest root-outermost.
2. Compute the common-prefix length `cp` between the currently-open `active` stack
   and `new_stack` by walking from the root while IPs match.
3. Close every open frame BELOW the divergence point (`active.size()` down to
   `cp`), emitting a slice `{start_ts .. this sample's ts}` for each as it pops.
4. Open every new frame from `cp` to the leaf, recording each with this sample's
   ts as its `start_ts`.

So a frame that persists across consecutive samples stays open (it is in the
common prefix every time) and produces ONE long slice, not one per sample. This
is the same prefix-diff idea the streaming `temporal_diff_filter` uses (doc 03,
section 3), applied here to timeline spans instead of trie interning - and it is
exactly why `coalesce_frames_by_name` matters (section 3.5): the diff is by IP, so
unstable per-block IPs would defeat it.

After the last sample, any still-open frames are closed at
`last_ts + sample_period_ns` (the one-period extension). Each close calls
`emit_close`, which resolves the frame's name (`resolve_name`) and emits the
`emitted_slice`.

### 5.4 `emit_nested_inline` - diffing on names, expanding inlines

When `inline_resolver` is set, `emit_nested` delegates to `emit_nested_inline`,
which is structurally the same algorithm with two differences:

- Each sampled IP is expanded via the resolver into its innermost-first chain,
  and the chain is pushed OUTERMOST-first into `new_stack` so the root inline frame
  becomes the outer slice. An IP whose chain comes back empty falls back to a
  single hex name (`resolve_name`).
- The open-frame stack holds NAMES (`open_frame{string name, start_ts}`) and the
  common-prefix diff compares names, not IPs.

Diffing on names means two consecutive samples whose leaf IPs differ but resolve
to the same function keep the frame open - the same coalescing
`coalesce_frames_by_name` provides for the IP path, but achieved inline, plus the
bonus that each inlined sub-frame becomes its own nested slice. 

### 5.5 Helpers (in `slice_emitter.cpp`)

`resolve_name(ip)` is the resolver wrapper that catches throws and falls back to
a hex name; `resolves_to_same_symbol(a, b)` wraps `symbol_identity`.

`filter_library_frames(smp)` (a private member function, not a nested class) is
what `process()` calls when `library_frame_filter` is set: it compacts a
`raw_sample` in place, keeping only surviving frames, resizing `frames`, and
resetting `smp.ip` to the new leaf. It holds no state of its own beyond the
callback in `m_cfg` - it does not own a `proc_maps` or any other module source
(unlike `sample_postprocessor`'s equivalent, which does, since `symbols` is
allowed to depend on itself). Filtering before shaping (rather than during) means
the diff and the slice spans are computed on the already-pruned stack, so dropped
library frames do not create spurious open/close churn.

---

## 6. The full offline flow, end to end

Putting the four components together, the canonical render (from the demos under
`examples/`) is:

```
samples = reconstructor.samples()           // per-tid map<tid, vector<raw_sample>>
                                            // (rebuilt from the sink's bytes)

// 1. symbolize in bulk
symbolizer sym(cfg);                        // include_binary_name / inline_frames
collect every unique IP across all samples
sym.preload(unique_ips);                    // one addr2line per binary

// 2a. (optional) coalesce IPs by name for the IP-diff path
coalesce_frames_by_name(samples, sym);      // OR set inline_resolver below

// 2b. (optional) filter + deliver named frames
sample_postprocessor pp({ .resolver = sym, .drop_library_frames = true, ... });
pp.process_all(samples, [](const processed_sample& s){ /* folded stacks, etc. */ });

// 3. shape the timeline and emit to a sink
slice_emitter_config scfg{ .resolver = ..., .inline_resolver = ... };
slice_emitter em(scfg);
em.process(samples, [&](const emitted_slice& s){ writer.write_slice(s); });
writer.finalize();                          // sink turns emitted_slice -> .pftrace
```

A consumer rarely uses both the postprocessor and the slice emitter on the same
run - they are two different output destinations. A flamegraph-collapsed or pprof
target reaches for the postprocessor's `processed_sample`s; a Perfetto-style
timeline target reaches for the emitter's `emitted_slice`s. Both share the
symbolizer (as the resolver/inline_resolver source) and, when they drop
library frames, the same `frame_filters` predicate. That sharing - one symbolizer,
one filter, two render targets - is the shape the four-way split is designed to
produce.

---

## 7. Invariants and pitfalls for a maintainer

- Borrowed views die with the call. `processed_sample::frames`,
  `emitted_slice::name`, and
  `symbolizer::resolve` / `file_line` results all reference component-owned
  storage. Copy out anything retained past the sink call (or past the
  symbolizer's lifetime / a `clear_cache()`).
- `frame_filters` predicates outlive nothing. The captured `proc_maps&` MUST
  outlive the predicate; both consumers own the maps via `unique_ptr` precisely
  for this. Do not hand a filter a temporary maps.
- IP-diff nested emission needs stable IPs. Run `coalesce_frames_by_name` before
  IP-based `emit_nested`, OR use `inline_resolver` (name diff). Skipping both
  re-introduces the one-slice-per-sample explosion (section 3.5).
- Symbolization is cold, and it shells out. `preload` is one subprocess per
  binary; `operator()` on a miss is one subprocess per IP. Always bulk-preload
  before rendering a large trace; never call the per-IP fallback in a tight loop.
- Resolver throws are swallowed everywhere (postprocessor, emitter,
  coalesce). A failing resolver degrades to an anonymous / hex frame, never an
  aborted pass - matching the best-effort philosophy of the wire side (doc 03,
  section 6).
- The render side is backend-neutral by construction. Nothing here knows whether
  a sample came from perf, the signal backend, or a replayed stream. Keep it that
  way: new symbol or predicate logic goes in the postprocessor/symbolizer lane,
  new output formats go behind `emitted_slice` in the emitter/sink lane, and
  neither lane should grow a dependency on capture (doc 06, section 13).
```