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

## Running vLLM Workload

A vLLM benchmark configuration is provided in `vllm.yaml` to measure profiler overhead on LLM inference workloads.

**Note:** The model path is currently fixed to `/model/Qwen/Qwen3-30B-A3B` in the YAML file. To use a different model, edit the `--model` parameter in `vllm.yaml` before running.

### Prerequisites
- vLLM installed and in PATH
- Model weights available at the specified path (or update path in `vllm.yaml`)
- Sufficient GPU memory for the model with tensor parallelism
