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
