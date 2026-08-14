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
