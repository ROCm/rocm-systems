# ML API tracing

Code for `--trace-ml-api`, which records PyTorch and Triton operators and the
GPU kernels they launch, and for the analyze options that read the result.

## Layout

- `constants.py` and `marker_format.py` are used by both modes. Recording writes
  the marker format, and analyze reads it.
- `profiling/` is used when profiling. It uses only the standard library.
  - `trace_selection.py` resolves the frameworks that `--trace-ml-api` selects.
- `analyze/` is used when analyzing. It can use pandas.
  - `definitions.py` holds the call-tree and trace-pair records.
  - `errors.py` holds the trace analysis errors.
  - `operator_patterns.py` matches `--list-operators` and `--operator` to the
    frameworks a workload traced.

## Import rule

A subpackage imports only from the top level of this package. It never imports
from the other subpackage, so pandas cannot reach profile mode.
