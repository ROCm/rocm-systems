# Marker API samples

These examples consume decoded marker records from the public
`rocprof_trace_decoder` Python API. They do not parse shaderdata JSON or extract
`.sqtt_funcmap` sections themselves. The shared loader uses the record filter to
request only marker and diagnostic records.

Run from the project root with the Python package on `PYTHONPATH`:

```bash
PYTHONPATH=python python3 samples/markers/print_markers.py \
  captures/*.att captures/*_code_object_id_*.out
```

Available examples:

- `print_markers.py` prints marker headers and payloads.
- `flamegraph.py` writes folded, SVG, or speedscope stack output.
- `payloads.py` exports marker payload records as JSON.
- `address_trace.py` reconstructs per-lane addresses and EXEC masks.
- `perfetto.py` writes filtered marker scopes and points as Perfetto JSON.
- `counter_fit.py` attributes SPM counter activity (for example
  `SQC_LDS_BANK_CONFLICT`) to source lines with a least-squares fit over SPM
  sample windows (Y = A.X), using address markers for the per-line execution
  counts. Any SPM counter can be passed with `--counter`. Without SPM input it
  runs a self-check against an LDS bank model. `lds_conflict.hip` is a test
  kernel with one conflict-free and one fully conflicting LDS load.
- `sqtt_site_rewrite.py` adds site markers to an already-compiled code object,
  so kernels can be profiled with `counter_fit.py` without rebuilding them with
  the sqtt-marker plugin. A `.sqtt_funcmap` section maps marker IDs to source
  lines. Works for gfx9 (all markers through `M0`) and gfx10+
  (`s_ttracedata_imm` for IDs 1..63, given to the sites in the most deeply
  nested loops; `M0` above that). `M0` is saved in an SGPR above every SGPR
  the code names (SGPR allocation raised on gfx9 if needed) or, when there is
  none, in an SGPR that register liveness shows is dead at the site.
  - In place (default): each LDS instruction becomes an `s_branch` to a stub
    (marker, the original instruction, `s_branch` back). Stubs go in unused
    padding between functions or after `.text`; out-of-reach stubs are reached
    through hops in padding. Later segments move when every PC-relative use
    can be adjusted.
  - Relayout (`--relayout`, and automatically when stubs are out of reach): a
    kernel is laid out again after `.text` with each marker inline before its
    instruction; branches and PC-relative constants are re-aimed (through
    branch islands when a branch ends up out of reach) and the kernel
    descriptor points at the new code. An `s_getpc` whose value is used
    in ways that cannot be re-aimed is pinned to the original code. If a
    kernel cannot be relaid out, an unchanged copy marked with stubs is used.
  - Per block (`--per-block`): one marker per basic block, before its first
    LDS instruction, instead of one per LDS instruction. Instructions in a
    block always run together, so the counts are the same; the funcmap row
    `site_lds_block@<lines>` lists every line in the block, and
    `counter_fit.py` attributes to the block. Much cheaper in hot loops (MIOpen
    Winograd: no measurable slowdown).
  - Only instructions reachable from kernel and function entries are marked
    (computed jumps built from `s_getpc` plus a constant are followed), so data
    stored in `.text` is never patched. Original code is never moved; code
    that still reaches it runs correctly, unmarked.
  - Before writing, the output is checked (every changed byte is planned, every
    branch chain and PC-relative address lands where planned, ELF layout and
    kernel descriptors are consistent, LLVM tools read it cleanly). Exit status
    3 or 4 means the code object was left unchanged.

  `lds_module.cpp` loads a code object with `hipModuleLoad` and dumps the
  kernel output for bit-for-bit comparison.
- `sqtt_site_injector.cpp` is a rocprofiler-sdk tool that runs
  `sqtt_site_rewrite.py` on every GPU code object as it is loaded, by wrapping
  `hsa_code_object_reader_create_from_memory` and `_from_file`, so unmodified
  binaries can be profiled. Set `ROCP_TOOL_LIBRARIES` to the built library and
  `SQTT_SITE_REWRITE` to the script; `SQTT_SITE_ARGS="--kernel <regex>"`
  limits rewriting to matching kernels. The original code object is loaded
  whenever the rewriter fails, finds nothing to mark, exceeds
  `SQTT_SITE_TIMEOUT` seconds (default 600; a 60 MB library takes about 3
  minutes and 5 GB), or the runtime rejects the
  rewritten object. It works alongside rocprofv3, which appends its own tool
  library.

The sample input parser derives nonzero code-object IDs from common rocprofiler
capture filenames such as `code_object_id_7.out`. Each code object is loaded
into the decoder so marker funcmaps are available during rolling trace parsing.

Each `Marker` contains one header and its attached `payload` words. Metadata
is shared across records from a parse. The scope samples reset their state at
`NEW_WAVE`; the reported hardware coordinates do not establish stable wave
identity across context save and restore (CWSR). The address example skips blocks
marked `INCOMPLETE` or whose observed payload count differs from the declaration.
Scope and point times use the marker timestamp in `shaderdata.time`.
Payload words are plain unsigned 32-bit integers. Subscribe to the separate
`SHADERDATA` record stream for original arrival timestamps and per-word flags.

The samples retain Python records for convenience. A callback consumer can use
`decoder.parse(..., on_batch=consume, collect=False)` to process batches without
retaining the full trace. Native C consumers can read borrowed contiguous
payload arrays directly during the callback.
