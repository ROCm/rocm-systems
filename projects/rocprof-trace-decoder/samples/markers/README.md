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
  the sqtt-marker plugin. Each LDS instruction is replaced by an `s_branch` to
  a stub (`s_ttracedata_imm <id>`, the original instruction, `s_branch` back)
  placed after `.text`, and a `.sqtt_funcmap` section maps IDs to source lines.
  Prototype limits: up to 63 sites per code object and stub space limited to
  the page slack after `.text`. `lds_module.cpp` loads a code object with
  `hipModuleLoad` and dumps the kernel output for bit-for-bit comparison.

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
