# Python API Samples

These samples use the public `rocprof_trace_decoder` Python API directly. They
take ATT files and code-object files as inputs, build the ISA provider in memory
from the code objects, decode the ATT buffers, and print or plot results.

Install the Python package first, or run from the source tree with
`PYTHONPATH=python`. Plotting samples require `matplotlib` and `numpy`.

The plotting samples write their PNG output to the current directory by
default. Use `-d DIR` to choose a different output directory.

The public API does not infer code-object IDs from filenames. These sample
scripts are command-line wrappers, so their input parser derives IDs from common
rocprofiler capture filenames before constructing explicit `CodeObject` values.
If exactly one code object has no inferred ID, these samples use ID `0`.
Multiple unnamed code objects are rejected because their IDs are ambiguous.

## ISA Hotspots

Print the top 30 instructions by `latency + idle`:

```bash
PYTHONPATH=python python3 samples/isa_hotspots.py \
  captures/*.att \
  captures/*_code_object_id_*.out
```

## Occupancy And Registers

Write `occupancy_resources.png` with active wave count plus active SGPR/VGPR
allocation over time:

```bash
PYTHONPATH=python python3 samples/plot_occupancy_resources.py \
  captures/*.att \
  captures/*_code_object_id_*.out
```

## Wave Lifetime

Write `wave_lifetime.png` plotting wave lifetime against `s_wait*` latency,
non-VALU latency, VALU latency, and idle time:

```bash
PYTHONPATH=python python3 samples/plot_wave_lifetime.py \
  captures/*.att \
  captures/*_code_object_id_*.out
```
