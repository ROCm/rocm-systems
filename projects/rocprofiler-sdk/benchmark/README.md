# Benchmark Suite

## Generate Data

From the current directory:

```shell
cmake -B build-benchmark .
cd build-benchmark
export PATH=${PWD}/bin:${PATH}
rocprofv3-benchmark -i ./example.yml -n 2
```

```shell
sqlite3 benchmark.db
```

```sql
SELECT * FROM benchmark_metrics;
SELECT * FROM benchmark_statistics;
```

## Measuring Kernel Replay Overhead

`kernel-replay.yaml` measures what kernel replay costs against the alternative it replaces.

Kernel replay collects N counter groups from one application run by re-executing each dispatch N
times, snapshotting and restoring device memory around each pass. Application replay runs the whole
application N times instead. Which is cheaper depends entirely on the workload, so this
configuration is built to find the crossover rather than to assert a number.

The jobs vary the three terms of the cost model — `dispatches x tracked footprint x passes` —
independently, so a result can be attributed to one of them instead of to the workload as a whole.
The headline comparison is between two named configurations collecting the same four counters:

```bash
rocprofv3-benchmark -i ./kernel-replay.yaml --filter-rocprofv3 replay_4_groups app_replay
```

One job is expected to show almost no overhead: `replay-hip-graph-declines`. A HIP graph launch is
not a single-packet single-dispatch submission, so replay declines it. That job exists to make the
absence visible here rather than in production, since a graph-capturing application gets no replay
at all.

See `docs/conceptual/kernel_replay/kernel_replay_performance.md` for what the numbers mean.

## Running vLLM Workload

A vLLM benchmark configuration is provided in `vllm.yaml` to measure profiler overhead on LLM inference workloads.

**Note:** The model path is currently fixed to `/model/Qwen/Qwen3-30B-A3B` in the YAML file. To use a different model, edit the `--model` parameter in `vllm.yaml` before running.

### Prerequisites
- vLLM installed and in PATH
- Model weights available at the specified path (or update path in `vllm.yaml`)
- Sufficient GPU memory for the model with tensor parallelism
