# RCCL Inspector Job Performance — Grafana Dashboard

This guide applies to **[`nccl-inspector-job-performance-template.json`](nccl-inspector-job-performance-template.json)** (dashboard title **RCCL Inspector Job Performance**, UID `rccl-inspector-job-performance-template`).

**Built-in help:** After import, open **Dashboard settings → General → Description**. Each variable also has a description under **Dashboard settings → Variables**. Grafana graphs sample values only; it does not display `# HELP` / `# TYPE`.

---

## Overview

Visualize RCCL collective and P2P performance from the [Inspector plugin](../README.md): bus bandwidth (GB/s) and execution time (µs) by collective, message size, rank count, and topology.

### Dashboard panels

| Row | Metrics shown |
|-----|---------------|
| RCCL Inspector - P2P [Recv] | P2P Recv bus bandwidth and exec time (intra-node and multi-node) |
| RCCL Inspector - P2P [Send] | P2P Send bus bandwidth and exec time (intra-node and multi-node) |
| RCCL Inspector - ReduceScatter | ReduceScatter bus bandwidth and exec time |
| RCCL Inspector - AllReduce | AllReduce bus bandwidth and exec time |
| RCCL Inspector - AllGather | AllGather bus bandwidth and exec time |

Each row splits into **intra-node / xGMI** (`n_nodes="1"`) and **network** (`n_nodes!="1"`).

### Dashboard variables

Set these **top to bottom**. There is no MySQL, Loki, or placeholder job metric.

| Variable | Type | Source |
|----------|------|--------|
| **Prometheus** (`prom_datasource`) | Datasource | The Prometheus (or Mimir / Thanos / VictoriaMetrics) datasource that scrapes Inspector textfiles. |
| **Cluster** (`cluster`) | Query | `label_values` on `nccl_bus_bandwidth_gbs` / `nccl_p2p_bus_bandwidth_gbs` → `cluster`. **All** is `.*` (`cluster=~"$cluster"`). |
| **Job ID** (`jobid`) | Query | Same series, label `slurm_job_id`, filtered by the selected cluster. Standalone `mpirun` is `unknown`. |
| **Node** (`node`) | Query | Hostnames for the selected cluster and job. Informational; panels are **not** filtered by node. |

---

## Prerequisites

### 1. RCCL Inspector exporting Prometheus metrics

```bash
export NCCL_PROFILER_PLUGIN=/path/to/librccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_PROM_DUMP=1
export NCCL_INSPECTOR_DUMP_DIR=/var/lib/node_exporter/nccl_inspector/
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=30000000  # 30s minimum
# export RCCL_DDA_ENABLE=0   # if DDA claims the collective and Inspector traces nothing
```

Point `NCCL_INSPECTOR_DUMP_DIR` at the node_exporter textfile collector directory. Files are `nccl_inspector_metrics_<gpu_uuid>.prom` and exist only while the job is running (they are unlinked at teardown when the dump thread is enabled).

### 2. Prometheus scraping the textfile directory

Configure node_exporter `--collector.textfile.directory=...` (or an equivalent) and scrape it. Expected series:

**Collective metrics**

| Metric | Description |
|--------|-------------|
| `nccl_bus_bandwidth_gbs` | Bus bandwidth in GB/s |
| `nccl_collective_exec_time_microseconds` | Execution time in µs |

**P2P metrics** (`NCCL_INSPECTOR_ENABLE_P2P=1`, the default)

| Metric | Description |
|--------|-------------|
| `nccl_p2p_bus_bandwidth_gbs` | P2P bus bandwidth in GB/s |
| `nccl_p2p_exec_time_microseconds` | P2P execution time in µs |

**Labels on every sample**

| Label | Description |
|-------|-------------|
| `cluster` | `NCCL_INSPECTOR_CLUSTER`, else `SLURM_CLUSTER_NAME`, else `unknown` |
| `slurm_job_id` | `SLURM_JOB_ID`, then `SLURM_JOBID`, `PBS_JOBID`, `LSB_JOBID`, else `unknown` |
| `n_nodes` | `"1"` = intra-node, `>1` = multi-node |
| `nranks` | Total ranks |
| `message_size` | Bucketed range (for example `8-9MB`) |
| `node`, `gpu`, `comm_name`, `version` | Host, HIP device index (`GPU3` = `hipGetDevice()==3`), communicator name, Inspector Prometheus format (`v5.1`, not the RCCL version) |

Collectives also have `collective` and `algo_proto`.

**Additional labels on P2P metrics:**

| Label | Description |
|-------|-------------|
| `p2p_operation` | `Send` or `Recv` |

### 3. Grafana

- Grafana with permission to import dashboards and manage datasources.
- The Prometheus instance above must be configured as a Grafana datasource.

---

## Import the dashboard

1. In Grafana, open **Dashboards** → **New** → **Import** (or **+** → **Import**).
2. Either:
   - **Upload JSON file** and select `nccl-inspector-job-performance-template.json`, or
   - Paste the file contents into **Import via panel json**.
3. Choose a **folder** (optional) and click **Import**.

If Grafana reports missing datasources, add or map them in **Connections** → **Data sources** first, then re-import or fix the datasource dropdowns on the affected panels.

---

## Using the dashboard

1. Confirm Explore (or Prometheus `/graph`) returns `nccl_bus_bandwidth_gbs`.
2. Open **RCCL Inspector Job Performance**.
3. Set **Prometheus**, then **Cluster**, then **Job ID**.
4. Time range must cover scrapes while the job was running.

| Job | Expect data | Expect empty |
|-----|-------------|--------------|
| Single-node AllReduce | AllReduce **intra-node (xGMI)** | AllReduce **NET**; P2P unless you ran Send/Recv |
| Multi-node AllReduce | AllReduce **NET** (`n_nodes!="1"`) | intra-node AllReduce; P2P unless Send/Recv |

`promtool check metrics < file.prom` may warn that `_microseconds` is not the Prometheus base unit `seconds`; the dashboard queries `_microseconds` on purpose.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| Cluster / Job ID dropdowns empty | Prometheus never scraped `.prom` files (dump dir empty, job already torn down, or node_exporter textfile path wrong) |
| Dropdowns populate, panels blank | Time range missed the scrapes, or you are on **NET** vs **xGMI** for the wrong `n_nodes` |
| No data in P2P panels | Job had no Send/Recv, or `NCCL_INSPECTOR_ENABLE_P2P=0` |
| Intra-node empty, NET has data | Multi-node job; expected |
| NET empty, intra-node has data | Single-node job; expected |
| `prom_datasource` missing | Add a Prometheus datasource in **Connections** → **Data sources** |
| Values look low vs rccl-tests | Inspector times GPU kernels; rccl-tests uses a host timer |

---

## Reuse and provisioning

- **UID** `rccl-inspector-job-performance-template` is fixed; change it only if it collides with an existing dashboard.
- For GitOps, register the JSON in Grafana [dashboard provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/#dashboards).
