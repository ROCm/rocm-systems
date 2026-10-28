# cpu-snap v1 wire format

This document specifies the on-the-wire byte format produced by the cpu-snap
streaming pipeline and consumed by the reader. The sampling sampler emits
this container format.

The format is a header-prefixed sequence of self-describing chunks. It is push-based: the producer hands completed byte spans to a host
`sink_fn`, and the consumer feeds arbitrary byte slices into a `reader` that
reassembles records. The library never opens a file descriptor; persistence is
the host's job.

For where this format sits in the larger design, see the sibling docs rather than
duplicating them here:

- [../README.md](../README.md) - component overview and build targets.

All multi-byte integers are little-endian. The producer
(`core::chunk_framer`) asserts a little-endian host at compile time; there is no
byte-swap path. The current format is **major 2, minor 0**
(`k_record_format_version_major` / `k_record_format_version_minor` in
`src/cpu_snap/core/records.hpp`).

## Source map

| Concern | File |
|---|---|
| Record structs, type tags, framing structs | `src/cpu_snap/core/records.hpp` |
| Producer (serialize records -> chunks) | `src/cpu_snap/core/chunk_framer.hpp` |
| Consumer (decode + resolve_stack) | `src/cpu_snap/core/reader.hpp`, `src/cpu_snap/core/reader.cpp` |
| Trie-diff compression (encode side) | `src/cpu_snap/sampling/streaming_pipeline.hpp`, `temporal_diff_filter.hpp`, `frame_trie.hpp` |
| Sink / event callback aliases | `src/cpu_snap/core/types.hpp` |

## Top-level stream layout

```
+--------------------------------------+
| file_header              (64 bytes)  |   emitted once, first
+--------------------------------------+
| chunk 0                              |   chunk_header + body
+--------------------------------------+
| chunk 1                              |
+--------------------------------------+
| ...                                  |
+--------------------------------------+
| chunk N (contains the EOF record)    |
+--------------------------------------+
```

The `file_header` is emitted exactly once by `chunk_framer::start()`. Every
subsequent emission is a complete chunk. The producer hands each of these (the
64-byte header, then each full chunk) to the host `sink_fn` as one call. The
consumer does not require any particular slicing: `reader::feed()` buffers
partial headers and bodies across calls.

## File header (64 bytes, packed)

Defined as `core::file_header`, declared with `#pragma pack(1)` so the struct
layout *is* the wire layout, and guarded by `static_assert(sizeof(file_header)
== 64)`.

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0  | 4  | `magic`         | ASCII `'C' 'S' 'N' 'P'` |
| 4  | 2  | `version_major` | = 2 |
| 6  | 2  | `version_minor` | = 0 |
| 8  | 8  | `start_unix_ns` | `CLOCK_REALTIME` nanoseconds since the Unix epoch |
| 16 | 8  | `pid`           | Linux pid of the recorded process |
| 24 | 4  | `chunk_bytes`   | Target chunk size the writer used, in bytes |
| 28 | 4  | `flags`         | bit 0 reserved (unused by current writers); bit 1 = frame-pointer stack; bit 2 = DWARF-unwound stack; bits 3-4 = stream clock (0 = `CLOCK_MONOTONIC`, 1 = `CLOCK_BOOTTIME`, 2 = `CLOCK_REALTIME`, 3 = reserved; a reader treats 3 as monotonic); other bits reserved (0) |
| 32 | 32 | `process_name`  | UTF-8, zero-padded; NOT guaranteed null-terminated if the name fills the buffer |

`magic == 'CSNP'` and `version_major == 2` are both mandatory. The reader treats
a magic mismatch or an unsupported major version as a **fatal** error: it sets an
internal fatal flag, bumps `parse_errors`, and silences all further emissions
(`reader::is_fatal()` then returns true).

## Chunk framing

A chunk is `chunk_header | records body`. The default target chunk size is 1 MiB
(`k_default_chunk_bytes = 1 << 20`, in `src/core/types.hpp`). The producer holds a
single inline `alignas(64)` 1 MiB buffer; `chunk_bytes` is a soft flush
threshold in the closed range `[k_min_chunk_bytes, k_default_chunk_bytes]`,
where `k_min_chunk_bytes = sizeof(chunk_header) + 4`.

Framing invariants enforced by `chunk_framer`:

- **Records never span a chunk.** If appending the next record (its padded size)
  would push `chunk_used + record_total` past `chunk_bytes`, the current chunk is
  sealed and emitted, and a fresh chunk is opened first.
- **Inter-record alignment is 4 bytes.** Each record is padded with zero bytes up
  to a 4-byte boundary (`align4`). The trailing pad of the last record is NOT
  counted in `records_byte_count`.

### Chunk header (16 bytes, packed)

`core::chunk_header`, `static_assert(sizeof == 16)`.

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0  | 2 | `magic`              | ASCII `'C' 'H'` |
| 2  | 2 | `version`            | chunk-header version, currently 0 |
| 4  | 4 | `records_byte_count` | unpadded body length in bytes (excludes the last record's trailing pad) |
| 8  | 8 | `chunk_seq`          | monotonic sequence number; gaps signal dropped chunks |

## Record framing inside a chunk body

The body is a packed sequence of records. Each record is a 4-byte header
followed by a body, then zero padding to the next 4-byte boundary.

Record header (4 bytes, `chunk_framer::record_header_wire`):

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 1 | `type`      | `record_type` tag (see below) |
| 1 | 1 | `reserved`  | always 0 |
| 2 | 2 | `body_size` | unpadded body length, u16; max `0xFFFF` (`k_max_record_body_bytes`) |

After the header come `body_size` bytes of body, then
`align4(4 + body_size) - (4 + body_size)` zero pad bytes. Record bodies are
serialized **field by field** in little-endian order; the in-memory record
structs are NOT packed, so on-wire layout is independent of host struct padding.
(The packed structs are only the framing ones: `file_header`, `chunk_header`.)

### Record types

`core::record_type` (u8 tag). Unknown tags (`0x08`..`0xFF` are reserved) are not
silently skippable record-by-record in this version: on an unknown or `invalid`
tag the reader's `record_walker` returns a parse error and abandons the rest of
the current chunk body.

| Code | Tag | Body | Section |
|---:|---|---|---|
| 0x00 | `invalid`    | -                | (never emitted) |
| 0x01 | `new_node`   | fixed, 16 bytes  | [NEW_NODE](#new_node-0x01) |
| 0x02 | `sample`     | fixed, 16 bytes  | [SAMPLE](#sample-0x02) |
| 0x03 | `checkpoint` | variable         | [CHECKPOINT](#checkpoint-0x03) |
| 0x04 | `maps_init`  | variable         | [MAPS_INIT](#maps_init-0x04) |
| 0x05 | `maps_delta` | variable         | [MAPS_DELTA](#maps_delta-0x05) |
| 0x06 | `lost`       | fixed, 16 bytes  | [LOST](#lost-0x06) |
| 0x07 | `eof`        | fixed, 16 bytes  | [EOF](#eof-0x07) |

The fixed-size bodies are all 16 bytes; the reader rejects them (returns a parse
error for that record) if `body_size != 16`.

## Trie-diff compression: NEW_NODE + SAMPLE

This is the core size win and the reason a single sample usually costs one
`sample_record` (16 bytes). Rather than writing every IP of every stack, the
encode side interns stacks into a grow-only trie and writes only the leaf id.

### What an IP in a stack means

Frames are stored innermost-first. The first frame is the interrupted
instruction, exact. Every later frame is a **call site**: the return address
minus one, so it lies inside the `call` instruction and symbol and line lookup
attribute the frame to the call, not to the instruction after it. All unwinders
(`kernel_callchain_unwinder`, `fp_copy_unwinder`, `dwarf_unwinder`) apply this
at capture time; consumers must not subtract again.

### How the encode side builds it

The compression lives in `sampling::internal::streaming_pipeline`, which
composes `temporal_diff_filter` over a shared `frame_trie`:

- `frame_trie` (`frame_trie.hpp`) is a Flyweight trie keyed on
  `(parent_node_id, ip)` edges. Each unique edge maps to one `node_id` (u32).
  Node id 0 is the implicit root; the first real node is id 1. `intern()` returns
  `{ id, was_new }`.
- `temporal_diff_filter` (`temporal_diff_filter.hpp`) keeps, per tid, the prior
  sample's root-ward path. For a new sample it computes the longest common
  root-ward prefix against that path and interns only the divergent suffix. A
  sample whose stack matches the previous sample on the same tid costs zero trie
  ops, zero allocations, and produces zero NEW_NODE records. `diff()` returns the
  `leaf_id` plus a span of nodes newly allocated on this call.

Frames are passed innermost-first (frames[0] = leaf, frames[n-1] = outermost),
matching `raw_sample` order, but the trie is built root-ward: a node's parent is
the frame closer to `main`. So a node's parent chain walks leaf -> ... -> root,
and the SAMPLE references the leaf node.

### The ordering contract

NEW_NODE records for the newly allocated nodes are ALWAYS emitted to the framer
**before** the SAMPLE that references the resulting `leaf_node_id`. Within one
sample's batch of new nodes the order is root-ward (parents before children).
This lets a forward-replay reader rebuild the trie and resolve any leaf id it
sees: every node is declared before it is referenced. A reader that encounters a
SAMPLE naming an unknown leaf resolves it to an empty stack rather than failing.

If the `trie_max_nodes` cap (default 2^22 nodes; 0 = unbounded) would be exceeded,
`frame_trie::intern()` throws; v1 has no raw-IP fallback record (deferred to v2).
The pipeline counts the sample in `internal_errors` and `dropped_samples`, and
emits a LOST record (`tid` = the sampled tid, `lost_count_delta` = 1).

When the sink throws for a chunk the framer discards it, which can lose NEW_NODE
records. The pipeline then replays every trie node as NEW_NODE records; a reader
treats a redefinition identical to the stored node as a no-op. A NEW_NODE naming
an unknown parent is rejected (counted in `parse_errors`) without abandoning the
rest of the chunk.

### NEW_NODE (0x01)

Body 16 bytes, fields in this order:

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 4 | `node_id`   | id assigned to this node; strictly positive (0 = root) |
| 4 | 4 | `parent_id` | parent node id; 0 = child of root |
| 8 | 8 | `ip`        | instruction pointer this frame represents |

### SAMPLE (0x02)

Body 16 bytes:

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0  | 8 | `timestamp_ns` | sample timestamp, nanoseconds of the stream clock (`flags` bits 3-4) |
| 8  | 4 | `tid`          | Linux tid of the sampled thread |
| 12 | 4 | `leaf_node_id` | trie node id of the innermost (leaf) frame |

## Other record bodies

### LOST (0x06)

Body 16 bytes. Mirrors a kernel `PERF_RECORD_LOST` count delta. The streaming
pipeline emits LOST with `tid == 0` (aggregate, not per-tid attributable).

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0  | 8 | `wall_ns`          | timestamp of the report, ns of the stream clock |
| 8  | 4 | `tid`              | 0 = aggregate / unknown |
| 12 | 4 | `lost_count_delta` | samples dropped since the previous report |

### EOF (0x07)

Body 16 bytes. Always the last record in the last chunk of a session; emitted by
`streaming_pipeline::finalize()` (which also runs from the destructor if not
called explicitly).

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 8 | `wall_ns`               | timestamp at session end, ns of the stream clock |
| 8 | 8 | `total_samples_emitted` | cumulative SAMPLE count for the session |

### CHECKPOINT (0x03)

Variable body. A periodic per-tid snapshot that lets a reader resync trie/counter
state after a missing or corrupt chunk. Emitted on a sample-count cadence
(`checkpoint_interval`, default 65536) and once at finalize.

Body layout:

```
u64 wall_ns
u32 n_entries
u32 reserved   (= 0)
n_entries x per_tid_checkpoint_entry (24 bytes each)
```

`per_tid_checkpoint_entry` (`k_per_tid_checkpoint_entry_bytes == 24`), in order:

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0  | 4 | `tid`     | Linux tid this row summarizes |
| 4  | 4 | `leaf_id` | most recently observed leaf node id for this tid |
| 8  | 8 | `samples` | cumulative SAMPLE count for this tid |
| 16 | 8 | `lost`    | cumulative LOST count attributed to this tid |

The reader rejects the record unless
`body_size == 16 + n_entries * 24` exactly. A checkpoint with more entries than
fit in one record (2730) is written as consecutive CHECKPOINT records sharing the
same `wall_ns` and carrying disjoint slices of the tids.

### MAPS_INIT (0x04)

Variable body. Full loaded-module list at session start, produced by
`sampling::dl_module_watcher` and emitted right after the file header. A list too
large for one record body is split across consecutive MAPS_INIT records, which a
reader applies additively. Body layout:

```
u64 wall_ns
u32 n_modules
u32 reserved   (= 0)
n_modules x module sub-record
```

Each **module sub-record** is:

| Field | Size | Notes |
|---|---:|---|
| `base_va`      | 8 | start of the module's executable mapping (page aligned), not the load bias |
| `size`         | 8 | span (bytes) of the module's executable mappings |
| `file_offset`  | 8 | ELF file offset of that mapping (page aligned) |
| `build_id_len` | 4 | length of `build_id` (writer caps at 64) |
| `path_len`     | 4 | length of `path` (writer caps at 4096) |
| `build_id`     | `build_id_len` | raw GNU build-id bytes |
| `path`         | `path_len` | absolute filesystem path of the backing ELF (no NUL terminator) |

The sub-record is then zero-padded to a 4-byte boundary. The fixed prefix is 32
bytes (`3*u64 + 2*u32`). The reader decodes modules sequentially via
`decode_module` and requires the cursor to land exactly on the body end.

### MAPS_DELTA (0x05)

Variable body. One module added or removed mid-stream. Body layout:

```
u64 wall_ns
u8  op           (0 = add, 1 = remove)
u8[3] op-padding (= 0)
u32 reserved     (= 0)
1 x module sub-record   (same layout as MAPS_INIT)
```

The fixed prefix is 16 bytes (`u64 + 4 (op + 3 pad) + u32`); the reader requires
the cursor to land exactly on the body end after the single module.

**Clocks.** Every timestamp in a stream (SAMPLE `timestamp_ns` and the `wall_ns`
of MAPS_INIT, MAPS_DELTA, LOST, CHECKPOINT and EOF) is in one clock, the stream
clock selected by `sampler::config::clock` and recorded in `file_header.flags`
bits 3-4; the default is `CLOCK_MONOTONIC`. A module's load time can therefore be
compared with a sample's timestamp (`module_map::resolve` does). Change from
earlier writers: LOST, CHECKPOINT and EOF `wall_ns` used to be `CLOCK_REALTIME`
(unix) ns regardless of the sample clock; they now use the stream clock. Files
written before this change have flag bits 3-4 = 0 (monotonic) and keep their
old LOST/CHECKPOINT/EOF values, which are unix ns; readers cannot tell the two
apart and should not compare those three fields with sample timestamps. The
header's `start_unix_ns` is always `CLOCK_REALTIME` and there is no conversion
anchor: with the `wallclock` stream clock, timestamps are unix ns directly. A module added between two polls is stamped with the previous
observation, so samples taken after the real `dlopen` resolve; a removal is
stamped with the poll time.

## Reader: decode and stack resolution

`core::reader` (`reader.hpp` + `reader.cpp`) is the symmetric counterpart of
`chunk_framer`. It is push-based and absorbing: `feed(span)` is the sole mutating
entry point and **never throws**. Parse errors, unknown record types, and
host-callback exceptions are all folded into `reader_counters` instead
of propagating.

### Internal collaborators

The implementation (anonymous namespace in `reader.cpp`) is three pieces:

- **`chunk_assembler`** - the framing state machine:
  `expect_file_header -> expect_chunk_header -> expect_chunk_body`, then back to
  `expect_chunk_header`. It accumulates exactly the next expected byte count into
  a pending buffer (header) or the body buffer, and on a complete chunk body
  hands the body span to the record walker.
- **`record_walker`** - walks one verified body, reading each 4-byte record
  header, bounds-checking `body_size`, decoding the body, dispatching to the host
  `event_fn`, and advancing by `align4(4 + body_size)`.
- **`trie_resolver`** - a `node_id -> {parent_id, ip}` map populated from every
  decoded NEW_NODE, used by `resolve_stack`.

### feed() error handling

- **Fatal**: bad file-header magic, unsupported `version_major`, or a
  `chunk_bytes` outside the reader's sane range. Sets the fatal flag, bumps
  `parse_errors`, and stops emitting (`is_fatal()` true thereafter).
- **Bad or implausible chunk header** (bad `'CH'` magic, bad `version`, a
  `records_byte_count` that would not fit inside the advertised
  `chunk_bytes`, or a `chunk_seq` that isn't a small forward step from the
  last one): bumps `parse_errors` (magic/version) or `chunks_rejected`
  (size/seq), then scans forward through the buffered bytes for the next
  plausible header (`resyncs`, `bytes_skipped`); this works across `feed()`
  call boundaries.
- **Unknown or reserved record type tag**: skipped by `body_size` and
  counted in `unknown_records`; the rest of the chunk is still walked
  (forward compatibility, per `records.hpp`).
- **Per-record errors on a known type** (size mismatch, truncated body,
  or a `NEW_NODE` that would create a cycle or redefine a node_id): bump
  `parse_errors` and abandon the rest of that chunk body.
- **Host callback exceptions**: caught and swallowed; not counted as stream
  errors.

`reader_counters` exposes `bytes_fed`, `records_emitted`, `parse_errors`,
`gap_count` (detected breaks in the `chunk_seq` run), `last_seen_seq`,
`chunks_rejected`, `unknown_records`, `resyncs`, and `bytes_skipped`.

### resolve_stack()

`reader::resolve_stack(leaf_id)` reconstructs a sample's IP chain from the
NEW_NODE trie built up so far. It walks parent pointers from `leaf_id` up to the
root (id 0), collecting one `ip` per node, and returns them innermost-first
(matching `raw_sample` / SAMPLE leaf-first order). If any node on the path is unknown it returns an empty vector. A typical consumer subscribes to
records via `on_event`, and for each `sample_record` calls
`resolve_stack(rec.leaf_node_id)` to get the frames, then symbolizes them
offline.
