# Performance dashboard

The Rocjitsu performance dashboard tracks **how quickly the simulator runs benchmark workloads**. It helps developers inspect benchmark history and identify changes in simulator performance.

!!! info "Simulator performance, not GPU hardware performance"

    These results measure Rocjitsu's simulation runs. They are not benchmarks of physical AMD GPUs and should not be used to compare AMD hardware performance. Target names such as `gfx950` identify the architecture being simulated.

[**Rocjitsu / Simulation Performance Dashboard** :material-open-in-new:](https://rocm.github.io/rocm-systems/rocjitsu-dashboard/){ .dashboard-launch aria-label="Open Rocjitsu Simulation Performance Dashboard (external site)" }

To benchmark a same-repository pull request targeting `develop`, add the [`rocjitsu-presubmit-benchmarks`](https://github.com/ROCm/rocm-systems/labels/rocjitsu-presubmit-benchmarks) label. The PR must match the workflow's [path filters](https://github.com/ROCm/rocm-systems/blob/develop/.github/workflows/rocjitsu-benchmarks.yml); fork PRs are excluded. Results from eligible runs are published to this dashboard.

## What you can explore

- Benchmark run history and test durations.
- Results for individual tests and simulated targets.
- Comparisons between a selected run and a baseline to investigate improvements or regressions.

## How to interpret the results

For a comparable, completed test, a shorter duration means the simulator finished that workload faster. Interpret changes alongside the test status, workload, simulator configuration, and benchmark environment. Failed or incomplete tests do not establish a performance improvement.

Use the dashboard to investigate simulator changes over time; use hardware benchmarks when evaluating performance on a physical GPU.
