# ML API tracing

Code for `--trace-ml-api`, which records PyTorch, Triton, and JAX operators and
the GPU kernels they launch, and for the analyze options that read the result.

## Layout

- `constants.py` and `marker_format.py` are used by both modes. Recording writes
  the marker format, and analyze reads it.
- `profiling/` is used when profiling. It uses only the standard library.
  - `trace_selection.py` resolves the frameworks that `--trace-ml-api` selects.
  - `orchestrator.py` checks the `--trace-ml-api` options and adds the launcher
    to the workload command.
  - `storage.py` copies the marker and counter files into the workload
    directory.
- `inject_roctx/` runs inside the workload while it is profiled. It does not
  use pandas.
  - `launch.py` is the entry point. It installs the selected backends and runs
    the workload script.
  - `installer.py` and `backend_contract.py` load and register the backends.
  - `marker_ranges.py` pushes and pops the ROCTX ranges.
  - `backends/` has one module for each framework, and `torch_wraps.py` lists
    what the torch backend wraps.
- `analyze/` is used when analyzing. It can use pandas.
  - `definitions.py` holds the call-tree and trace-pair records.
  - `errors.py` holds the trace analysis errors.
  - `orchestrator.py` runs the analysis steps in order for one workload.
  - `trace_csv.py` finds and reads the marker and counter files.
  - `pass_matching.py` joins markers and dispatches and matches operator calls
    across profiling passes.
  - `call_tree.py` nests the markers into call trees and adds up kernel stats.
  - `selection.py` finds the operators that match `--operator` and the
    dispatches they launched.
  - `output.py` prints the operator call tree, the matched-operator report and
    the operator summary table.
  - `operator_patterns.py` matches `--list-operators` and `--operator` to the
    frameworks a workload traced.

## Import rule

A subpackage imports only from the top level of this package. It never imports
from the other subpackage, so pandas cannot reach profile mode.
