# rocpd databases and tools

Contents:

1. [Database views](#database-views)
2. [SQL recipes](#sql-recipes)
3. [rocpd subcommands](#rocpd-subcommands)
4. [Multiple databases](#multiple-databases)

Every rocprofv3 mode writes a rocpd database (`<name>_results.db`, SQLite). Query it through the
views below rather than the UUID-suffixed tables. Timestamps and durations are nanoseconds on one
clock shared by host and GPU records. Python's `sqlite3` module or the `sqlite3` CLI is enough;
the scripts in this skill use only the views.

## Database views

| View | Useful columns |
| --- | --- |
| `kernels` | `name`, `region` (ROCTx name with `--kernel-rename`), `pid`, `tid`, `agent_abs_index`, `agent_type`, `dispatch_id`, `stream_id`, `queue_id`, `start`, `end`, `duration`, `grid_x/y/z` (work-items), `workgroup_x/y/z`, `lds_size`, `scratch_size`, `vgpr_count`, `accum_vgpr_count`, `sgpr_count`, `graph_exec_id`, `graph_node_id`, `corr_id` |
| `memory_copies` | `name` (for example `MEMORY_COPY_HOST_TO_DEVICE`), `size` (bytes), `start`, `end`, `duration`, `src_agent_type`, `dst_agent_type`, `src_agent_abs_index`, `dst_agent_abs_index`, `stream_id` |
| `regions` | Host API calls, ROCTx ranges, and KFD events: `category`, `name`, `pid`, `tid`, `start`, `end`, `duration`, `corr_id`, `extdata` |
| `memory_allocations`, `scratch_memory` | Allocation type, level, size, address, agent |
| `counters_collection` | One row per dispatch and counter: `dispatch_id`, `kernel_name`, `counter_name`, `value`, `duration`, grid, workgroup, registers, LDS, scratch |
| `spm_counters` | SPM samples with `timestamp`, `xcc`, `shader_engine`, `instance`, `value` |
| `processes`, `threads` | `pid`, `command`, `start`, `end` |
| `kernel_symbols`, `code_objects` | Kernel names (mangled, demangled, truncated), register counts, code object URIs |
| `rocpd_info_agent` | `absolute_index`, `type`, `type_index`, `name` (gfx target), `product_name`, `extdata` (JSON with `cu_count`, `wave_front_size`, `num_xcc`, ...) |
| `top_kernels` | `name`, `total_calls`, `total_duration` and `average` (microseconds), `percentage` |
| `top` | The same columns over kernels, copies, and regions together |
| `busy` | Per agent `GpuTime / WallTime`; sums durations, so it can exceed 1 when work overlaps |

Region categories from `--sys-trace` in rocprofv3 1.4.1: `HIP_RUNTIME_API_EXT`,
`HIP_COMPILER_API_EXT`, `HSA_CORE_API`, `HSA_AMD_EXT_API`, `MARKER_CORE_RANGE_API` (ROCTx ranges),
`KFD_PAGE_FAULT`, `KFD_PAGE_MIGRATE`, `KFD_EVENT_UNMAP_FROM_GPU`. ROCTx marks and RCCL calls use
`MARKER_*` and `RCCL_*` categories. Match on prefixes (`category LIKE 'HIP_%'`) because suffixes
vary between releases.

## SQL recipes

```sql
-- Kernel statistics
SELECT name, COUNT(*) AS calls, SUM(duration)/1e6 AS total_ms, AVG(duration)/1e3 AS avg_us
FROM kernels GROUP BY name ORDER BY total_ms DESC LIMIT 20;

-- Copy bandwidth by direction (bytes per ns equals GB/s)
SELECT name, COUNT(*), SUM(size)/1048576.0 AS mib, SUM(size)*1.0/SUM(duration) AS gbps
FROM memory_copies GROUP BY name;

-- Kernels inside one ROCTx range on the same thread
SELECT k.name, COUNT(*), SUM(k.duration)/1e6 AS ms
FROM regions r JOIN kernels k ON k.tid = r.tid AND k.start BETWEEN r.start AND r.end
WHERE r.category LIKE 'MARKER%' AND r.name = 'iteration' GROUP BY k.name;

-- Host API calls slower than 1 ms
SELECT name, start, duration/1e6 AS ms FROM regions
WHERE category LIKE 'HIP_%' AND duration > 1000000 ORDER BY duration DESC;

-- Mean counter values per kernel
SELECT kernel_name, counter_name, COUNT(*) AS dispatches, AVG(value) AS mean_value
FROM counters_collection GROUP BY kernel_name, counter_name ORDER BY kernel_name;
```

Kernel `start`/`end` are GPU execution times; host API calls return before the kernel starts, so
joins on time are approximate. Use `corr_id` to link a kernel to the API call that launched it.

## rocpd subcommands

`rocpd` (also `python3 -m rocpd`) works on one or many databases, directories, or `.rpdb`
packages.

| Command | Purpose | Needs pandas |
| --- | --- | --- |
| `rocpd convert -i DB -f pftrace` | Perfetto trace for ui.perfetto.dev | no |
| `rocpd convert -i DB -f otf2` | OTF2 for Vampir; better for very large traces | no |
| `rocpd convert -i DB -f csv` | `<out>_kernel_trace.csv`, `_memory_copy_trace.csv`, `_regions_trace.csv`, `_counter_collection_trace.csv`, `_agent_info.csv`, ... (the copy CSV has no byte sizes) | yes |
| `rocpd summary -i DB [--region-categories NONE] [--summary-by-rank] [--format csv html json]` | Kernel, copy, allocation, and API summaries; per-rank comparison | yes |
| `rocpd query -i DB --query "SQL" [--format csv html json md]` | Run SQL over one or many databases | yes |
| `rocpd merge -i A.db B.db -o merged` | Combine databases into one file | no |
| `rocpd package -i DIR -d name --consolidate --copy` | Bundle databases into `name.rpdb` with an `index.yaml` | no |

Shared convert options: `-d OUTPUT_PATH` (default `./rocpd-output-data`), `-o OUTPUT_FILE`,
`--kernel-rename`, `--group-by-queue`, `--agent-index-value absolute|relative|type-relative`,
`--start 25%|<ns>`, `--end 75%|<ns>`, `--start-marker NAME`, `--end-marker NAME`. Standalone
wrappers `rocpd2pftrace`, `rocpd2otf2`, `rocpd2csv`, and `rocpd2summary` take the same options.

If `rocpd` fails with a Python version mismatch, run it with the interpreter it was built for:
`python3.10 $(which rocpd) convert ...`.

## Multiple databases

- More than one input database is merged automatically into a temporary `.rpdb` unless
  `--automerge-limit N` (max 8) allows attaching them directly (SQLite attaches at most 10).
- `rocpd merge` writes one large file; `rocpd package` references databases in place, which is
  better for many large rank databases.
- For MPI runs, `rocpd summary -i run_*.db --summary-by-rank` compares ranks side by side.
