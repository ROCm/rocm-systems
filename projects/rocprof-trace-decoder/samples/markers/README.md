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
