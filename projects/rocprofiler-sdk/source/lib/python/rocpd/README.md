# ROCm Profiling Data (RocPD)

The RocPD Python package provides a scriptable API for analyzing, summarizing, filtering, and merging tracing data
collected with the ROCm profiling tools suite.

## Background

In the past, the ROCm profiling tools (e.g. rocprofv3, rocprofiler-systems, etc.) have directly written data to
various output formats such as CSV, JSON, Perfetto, OTF2, etc. This approach has a significant number of flaws:

### No standardization in the CSV and JSON output formats

The ROCm profiling groups consider the standardization of the CSV and JSON output formats for all the tools
as a waste of time. Neither of these data formats scale well when large amounts of profiling data is collected.
Due to the inherent overhead of parsing textual data as opposed to binary, the arcane simplicity of the
CSV format, and the (general) requirement to parse/load the entire JSON file in order to perform any meaningful
data processing.

### Inability to unify output collected across multiple processes and nodes

Supporting the unification of output collected across multiple processes and nodes is a difficult endeavor.
The complexity of communicating profiling information between processes, especially when the processes exist
on separate nodes connected through a network, at best, requires integration with the job launchers and/or
explicit support for the job launchers. The general expectation for profiling tools is for them to work
regardless of the user application's choice of process-level parallelism (e.g. MPI, fork, spawn,
Python multi-processing, UPC, etc.) and job scheduler (e.g. SLURM, flux, PBS/Torque, LSD, etc.).
Adding explicit integration/support for this many flavors of parallelism and jobs schedulers is untenable.
The most consistent aspect of multi-node jobs is a shared filesystem: it is considered a necessity for the
user experience. Without a shared filesystem, the user would be responsible for transferring the application's
input and output to/from the specific nodes the job scheduler decided to give them. Thus, the most reliable
output for in-process profiling tools is adopting the approach of generating (at least) one output file per process.

In order to unify the output collected across multiple processes, the one-output-per-process approach
requires either (A) a post-processing step which combines the various outputs into a single output,
(B) an output format which utilizes a single "metadata" file which links together the individual
outputs, or (C) a visualizer which supports opening multiple files at once. The ROCm profiling group
considers Option A as the most flexible and reliable approach since Option B does require a small
amount of inter-process communication to write the "metadata" file and Option C imposes a rigid
restriction on the choice of visualizer.

### Data filtering at the data collection stage

In rocprofiler-systems and rocprofv3 with the direct output to Perfetto approach, if the tool collects
2 GB of tracing data per-process in a multi-node job with 16 processes, Perfetto will struggle
to visualize each individual 2 GB trace and fail to load a combined 32 GB trace. In this situation, the
user must re-run the application and collect less data -- all of that tracing data from the previous run
is effectively lost. However, if rocprofiler-systems and rocprofv3 were to adopt an intermediate output
format approach and the Perfetto visualization is generated from this intermediate output format,
the user would have a multitude of options to remedy this issue. For example, the user could filter out
data (e.g. drop HSA functions from the trace), instruct the Perfetto generator to skip adding Perfetto
debug annotations on the trace events, combine the 32 GB of data and split it into 32 separate visualizations
based on time instead of processes, etc.

### Absence of automated analysis

Certain formats such as Perfetto are great for visualization. However, they lack any automated analysis
of the data. For example, a flat profile is an extremely useful companion when visually analyzing a trace
and other forms of automated analysis can quickly and easily do anomaly detection.

## Overview

RocPD is essentially a Python package which understands a standardized SQLite3 schema. This Python package
intends to provide a centralized place for a multitude of post-process analysis capabilities. The capabilities
include, but are not limited to, analyzing, summarizing, filtering, merging, and generating visualizations of
tracing data. This design allows tools such as rocprofv3, rocprofiler-systems, rocprofiler-compute, etc. to
focus on minimizing overhead during data collection and adding new data collection features. These tools simply
need to write one SQL database per process which adheres to the agreed upon RocPD SQL schema and RocPD will
handle the analysis and visualization of the data.

RocPD uses a unique approach to view multiple on-disk databases as a single-database when performing queries.
Python applications using RocPD __must__ load the on-disk databases by constructing a `rocpd.importer.RocpdImportData`
object with a list of the database filepaths or by using the `rocpd.connect` function which returns a
`rocpd.importer.RocpdImportData` object. For the schema and how to query the data, see the RocPD schema section below.

### Loading Databases Example

```python
input = ["A.db", "B.db"]
rpd_data = rocpd.connect(input)
```

### Executing Queries

The `rocpd.importer.RocpdImportData` object supports all of the same functions as `sqlite3.Connection`:

```python
for itr in rpd_data.execute("SELECT * FROM kernels"):
    print(f"{itr}")

cursor = rpd_data.cursor()
for itr in cursor.execute("SELECT * FROM top").fetchall():
    print(f"{itr}")
```

## RocPD schema

The database uses a normalized SQLite schema. Raw data lives in base tables; analytical views simplify querying by
joining tables and resolving IDs to human-readable names.

### Key concepts

**Table names and UUIDs.** On disk, each table has a UUID appended (e.g. `rocpd_sample_abc123`) so that table names
remain unique when multiple databases are combined. You should ignore the UUID and use only the common table names
(e.g. `rocpd_info_pmc`, `rocpd_sample`). The rocpd_views layer exposes these common names; when you query
`rocpd_sample`, you are querying the correct logical table regardless of the underlying UUID.

**Info tables vs. event tables.** The `rocpd_info_*` tables are informational (lookup) tables: they describe entities
such as nodes, processes, threads, agents, queues, streams, PMC definitions, code objects, and kernel symbols, and
are referenced by other tables via foreign keys. The other `rocpd_*` tables (e.g. `rocpd_region`, `rocpd_sample`,
`rocpd_kernel_dispatch`, `rocpd_memory_copy`) are event or repeated-data tables: they store the actual trace data
and reference the info tables and `rocpd_string` for names and metadata.

**Prefer common views over base tables.** The common views abstract away both the UUID naming and the need to write
complex JOINs. They expose human-readable columns (e.g. kernel name, duration, device names). You should query the
common views (e.g. `kernels`, `regions`, `samples`, `memory_copies`) for analysis rather than the base tables. Use
base table names only when you need to understand the schema or write custom JOINs.

### Schema reference

| Table | Purpose | Key columns (plain language) |
| ----- | ------- | --------------------------- |
| rocpd_metadata | Schema version and database identifier (tag/value pairs). | tag, value |
| rocpd_string | Deduplicated strings (names, categories); other tables reference by id. | id, string |
| rocpd_info_node | Machine or node (hostname, system name). | id, hostname, machine_id, system_name |
| rocpd_info_process | Process (OS pid, parent pid, command, start/end time). | id, nid, pid, ppid, command, start, end |
| rocpd_info_thread | Thread (OS tid, name); belongs to a process. | id, pid, tid, name, start, end |
| rocpd_info_agent | CPU or GPU agent (type, indices, name, model). | id, type, absolute_index, name, model_name |
| rocpd_info_queue | Queue metadata; referenced by kernel and memory events. | id, nid, pid, name |
| rocpd_info_stream | Stream metadata; referenced by kernel and memory events. | id, nid, pid, name |
| rocpd_info_pmc | Performance counter (PMC) definitions (name, symbol, description). | id, name, symbol, description, value_type |
| rocpd_info_code_object | Code object (URI, load base/size, storage type). | id, agent_id, uri, load_base, load_size, storage_type |
| rocpd_info_kernel_symbol | Kernel symbol (kernel name, segment sizes, vgpr/sgpr counts). | id, code_object_id, kernel_name, display_name, group_segment_size, private_segment_size |
| rocpd_track | Track for grouping samples (references name, process, thread). | id, name_id, pid, tid |
| rocpd_event | Event metadata (category, call stack, correlation id); shared by regions, samples, kernels. | id, category_id, call_stack, correlation_id |
| rocpd_arg | Event arguments (position, type, name, value). | id, event_id, position, type, name, value |
| rocpd_pmc_event | PMC value recorded for an event. | id, event_id, pmc_id, value |
| rocpd_region | CPU region with start and end time on one thread. | id, nid, pid, tid, start, end, name_id, event_id |
| rocpd_sample | Instantaneous sample (single timestamp). | id, track_id, timestamp, event_id |
| rocpd_kernel_dispatch | GPU kernel launch (agent, kernel, queue, stream, grid/workgroup sizes). | id, agent_id, kernel_id, queue_id, stream_id, start, end, grid_size_*, workgroup_size_* |
| rocpd_memory_copy | Memory copy operation (source/destination, size, time range). | id, start, end, size, name_id, src_agent_id, dst_agent_id |
| rocpd_memory_allocate | Memory allocation or free (type: ALLOC/FREE/REALLOC/RECLAIM; level: REAL/VIRTUAL/SCRATCH). | id, type, level, start, end, size, address, agent_id |

### Common views

The following views join base tables and resolve IDs to names so you can analyze data without writing JOINs. When
using RocpdImportData with multiple databases, these views query across the attached databases.

| View | Purpose | When to use |
| ---- | ------- | ----------- |
| processes | Process list with node info (hostname, machine_id, pid, command). | Filter by process or node. |
| threads | Thread list with process and node. | Per-thread analysis. |
| code_objects | Code objects with agent index and URI. | Code object or load info. |
| kernel_symbols | Kernel symbols with process; display and demangled names. | Kernel metadata. |
| regions | CPU regions with category, name (text), duration, start/end, call_stack, line_info. | CPU timelines and hotspots. |
| region_args | Arguments for region events. | API arguments tied to regions. |
| samples | Instant samples with category, name, timestamp, call_stack. | Sampling profiles. |
| sample_regions | Samples in the same column shape as regions. | Use with regions_and_samples. |
| regions_and_samples | Union of regions and sample_regions. | One query over all CPU activity. |
| kernels | Kernel dispatches with kernel name, agent, queue, stream, duration, grid/workgroup, LDS/scratch. | GPU kernel analysis. |
| memory_copies | Copies with duration, size, src/dst device names, region name. | Transfer analysis. |
| memory_allocations | Alloc/free with type, level, agent, size, queue/stream. | Memory usage. |
| scratch_memory | Scratch-only allocations (subset of memory_allocate). | GPU scratch analysis. |
| pmc_info | PMC counter metadata. | Discover available counters. |
| pmc_events | PMC values per kernel (counter name, value, duration). | Counter time series. |
| counters_collection | Aggregated PMC per dispatch/kernel/counter (e.g. SUM of value). | Counter summaries per kernel. |
| events_args | Events with argument rows (category, arg name/value). | API or event arguments. |
| stream_args | Stream arguments with stream description. | Stream-related API analysis. |

### Summary views

Using the common views above, a few default summary views are created for quick, high-level analysis of the data.

| View | Purpose | When to use |
| ---- | ------- | ----------- |
| busy | GPU utilization per agent: GpuTime (kernels + memory copies), WallTime (span of trace), and Busy = GpuTime/WallTime. | Check how busy each GPU was; spot underutilization. |
| top | Overall time breakdown across kernels, memory copies, and CPU regions: name, total_calls, total_duration (µs), average (µs), and percentage of total trace time. | Single view of “what used the most time” across GPU and CPU. |
| top_kernels | Sorted list of kernels by total GPU time: name, total_calls, total_duration (µs), average (µs), and percentage of all kernel time. | Find the most expensive kernels; compare relative cost. |

### Example queries

The following examples use the common views and assume timestamps and durations are in nanoseconds. You can run them
with `rpd_data.execute("SELECT ...")` or a cursor.

**Top 10 kernels by total time:**

```sql
SELECT name, SUM(duration) AS total_ns FROM kernels GROUP BY name ORDER BY total_ns DESC LIMIT 10;
```

**Memory copies larger than 1 MB:**

```sql
SELECT name, size, duration, src_device, dst_device FROM memory_copies WHERE size > 1048576 ORDER BY size DESC;
```

**CPU regions whose name contains a given string:**

```sql
SELECT name, category, duration, start, end FROM regions WHERE name LIKE '%foo%' ORDER BY start LIMIT 20;
```

**Kernel dispatches on a specific GPU (e.g. agent index 0):**

```sql
SELECT name, duration, grid_x, grid_y, grid_z FROM kernels WHERE agent_abs_index = 0 ORDER BY start;
```


### Detailed schema explanation

For each base table, the following lists every column and what it represents. Use the common table names (e.g.
`rocpd_region`) when querying; the UUID-suffixed names are for internal use.

**rocpd_metadata**

Purpose: Serves as a table for DBs to record metadata, especially for schema version and UUID key.  Can be extended to store additional metadata key value pairs.

| Field | Meaning |
| ----- | ------- |
| id | Auto-increment primary key. |
| tag | Key for a metadata entry (e.g. "schema_version", "uuid"). |
| value | Value for the tag (e.g. schema version number, UUID string). |

**rocpd_string**

Purpose: Serves as a string look-up table. Instead of embedding frequently used strings directly in text fields, strings are mapped to a unique string ID that other tables can reference.

| Field | Meaning |
| ----- | ------- |
| id | Auto-increment primary key; other tables reference this for names and categories. |
| guid | Database/import identifier. |
| string | The deduplicated string (e.g. region name, category name). Unique per guid. |

**rocpd_info_node**

Purpose: Serves as an info table to store the node identification information. 

| Field | Meaning |
| ----- | ------- |
| id | Primary key; referenced by process and other info tables as nid. |
| guid | Database/import identifier. |
| hash | Unique hash for the node. |
| machine_id | Unique machine identifier (i.e. taken from system's /etc/machine-id). |
| system_name | Operating system name. |
| hostname | Host name. |
| release | OS release string. |
| version | OS version. |
| hardware_name | Hardware name. |
| domain_name | Domain name. |

**rocpd_info_process**

Purpose: Serves as an info table to store process related information.

| Field | Meaning |
| ----- | ------- |
| id | Primary key; referenced by thread, agent, and event tables as pid. |
| guid | Database/import identifier. |
| nid | References rocpd_info_node.id (node this process runs on). |
| ppid | Parent process id (OS). |
| pid | Process id (OS). |
| init | Initialization timestamp when profiling started. |
| fini | Finalization timestamp when profiling completed. |
| start | Process start timestamp. |
| end | Process end timestamp. |
| command | The command being profiled (ideally includes the arguments provided, to be profiled). |
| environment | JSON object of environment variables. |
| extdata | JSON for any extra data. |

**rocpd_info_thread**

Purpose: Serves as an info table to store thread related information.

| Field | Meaning |
| ----- | ------- |
| id | Primary key; referenced by region, sample track, and event tables as tid. |
| guid | Database/import identifier. |
| nid | References rocpd_info_node.id. |
| ppid | Parent process id (OS). |
| pid | References rocpd_info_process.id (process this thread belongs to). |
| tid | Thread id (OS). |
| name | Thread name. |
| start | Thread start timestamp. |
| end | Thread end timestamp. |
| extdata | JSON for any extra data. |

**rocpd_info_agent**

Purpose: Serves as an info table for agents (CPUs, GPUs, and other devices). Each row includes absolute_index (global order), logical_index (runtime order), and type_index (order within CPU or GPU only) for labeling and filtering.

| Field | Meaning |
| ----- | ------- |
| id | Primary key; referenced by kernel_dispatch, memory_copy, etc. as agent_id. |
| guid | Database/import identifier. |
| nid | References rocpd_info_node.id. |
| pid | References rocpd_info_process.id. |
| type | One of: 'CPU', 'GPU'. |
| absolute_index | Absolute index of the agent. |
| logical_index | Logical index. |
| type_index | Index within the type (CPU or GPU). |
| uuid | Agent UUID. |
| name | Agent name. |
| model_name | Model name (e.g. GPU model). |
| vendor_name | Vendor name. |
| product_name | Product name (e.g. Marketing name - AMD Instinct MI300X). |
| user_name | User-visible name. |
| extdata | JSON for any extra data. |

**rocpd_info_queue**

Purpose: Serves as an info table for HSA queues (or HIP streams backed by HSA queues) on which kernel and memory-copy operations are enqueued.

| Field | Meaning |
| ----- | ------- |
| id | Primary key; referenced by kernel_dispatch, memory_copy, etc. as queue_id. |
| guid | Database/import identifier. |
| nid | References rocpd_info_node.id. |
| pid | References rocpd_info_process.id. |
| name | Queue name (e.g. Default Queue, Queue 1, Queue 2, etc). |
| extdata | JSON for any extra data. |

**rocpd_info_stream**

Purpose: Serves as an info table to describe the stream information. Each row corresponds to a HIP stream (the ordering context used by the HIP API) with which kernel dispatches and memory operations may be associated.

| Field | Meaning |
| ----- | ------- |
| id | Primary key; referenced by kernel_dispatch, memory_copy, etc. as stream_id. |
| guid | Database/import identifier. |
| nid | References rocpd_info_node.id. |
| pid | References rocpd_info_process.id. |
| name | Stream name. (e.g. Default Stream, Stream 1, Stream 2, etc.) |
| extdata | JSON for any extra data. |

**rocpd_info_pmc**

Purpose: Serves as an info table to describe performance counters. Each row corresponds to a description of a performance counter that was profiled.  Any actual values of the PMC would be recorded in the rocpd_pmc_event table.

| Field | Meaning |
| ----- | ------- |
| id | Primary key; referenced by rocpd_pmc_event.pmc_id. |
| guid | Database/import identifier. |
| nid | References rocpd_info_node.id. |
| pid | References rocpd_info_process.id. |
| agent_id | References rocpd_info_agent.id (agent this counter belongs to); may be null. |
| target_arch | One of: 'CPU', 'GPU'. |
| event_code | Hardware event code. |
| instance_id | Instance identifier. |
| name | Counter name. |
| symbol | Counter symbol (shorthand name). |
| description | Short description. |
| long_description | Long description. |
| component | Component name (source of counter: rocm, amd_smi, papi). |
| units | Units (e.g. bytes, cycles). |
| value_type | One of: 'ABS', 'ACCUM', 'RELATIVE'. |
| block | Block name. |
| expression | Expression for derived counters. |
| is_constant | 1 if constant, 0 otherwise. |
| is_derived | 1 if derived, 0 otherwise. |
| extdata | JSON for any extra data. |

**rocpd_info_code_object**

Purpose: Serves as an info table for GPU code objects (loaded kernel code and metadata per agent). Kernel symbols reference code_object_id to indicate which code object contains each kernel.

| Field | Meaning |
| ----- | ------- |
| id | Primary key; referenced by rocpd_info_kernel_symbol.code_object_id. |
| guid | Database/import identifier. |
| nid | References rocpd_info_node.id. |
| pid | References rocpd_info_process.id. |
| agent_id | References rocpd_info_agent.id. |
| uri | Code object URI (e.g. file path). |
| load_base | Load base address. |
| load_size | Load size. |
| load_delta | Load delta. |
| storage_type | One of: 'FILE', 'MEMORY'. |
| extdata | JSON for any extra data. |

**rocpd_info_kernel_symbol**

Purpose: Serves as an info table for kernel symbols (kernel name, display name, segment sizes, vgpr/sgpr counts). Each kernel dispatch references a row here via kernel_id to identify the kernel that was launched.

| Field | Meaning |
| ----- | ------- |
| id | Primary key; referenced by rocpd_kernel_dispatch.kernel_id. |
| guid | Database/import identifier. |
| nid | References rocpd_info_node.id. |
| pid | References rocpd_info_process.id. |
| code_object_id | References rocpd_info_code_object.id. |
| kernel_name | Kernel name (symbol). |
| display_name | Human-readable kernel name. |
| kernel_object | Kernel object handle. |
| kernarg_segment_size | Kernarg segment size. |
| kernarg_segment_alignment | Kernarg segment alignment. |
| group_segment_size | Group (LDS) segment size. |
| private_segment_size | Private (scratch) segment size. |
| sgpr_count | Number of scalar general-purpose registers. |
| arch_vgpr_count | Number of vector general-purpose registers. |
| accum_vgpr_count | Number of accumulator VGPRs. |
| extdata | JSON for extra data. |

**rocpd_track**

Purpose: Groups instantaneous samples into named tracks (each track has a name, process, and thread); rocpd_sample rows reference track_id to indicate which track the sample belongs to.

| Field | Meaning |
| ----- | ------- |
| id | Primary key; referenced by rocpd_sample.track_id. |
| guid | Database/import identifier. |
| nid | References rocpd_info_node.id. |
| pid | References rocpd_info_process.id (may be null). |
| tid | References rocpd_info_thread.id (may be null). |
| name_id | References rocpd_string.id for the track name. |
| extdata | JSON for any extra data. |

**rocpd_event**

Purpose: Shared event metadata (category, call stack, correlation id) referenced by regions, samples, kernels, and memory operations via event_id for correlation and call-stack analysis.

| Field | Meaning |
| ----- | ------- |
| id | Primary key; referenced by region, sample, kernel_dispatch, memory_copy, etc. as event_id. |
| guid | Database/import identifier. |
| category_id | References rocpd_string.id for the event category name. |
| stack_id | Stack identifier for call stack grouping. |
| parent_stack_id | Parent stack identifier. |
| correlation_id | Correlation id to link related events (e.g. API call and kernel). |
| call_stack | JSON array representing the call stack. |
| line_info | JSON with line/source info. |
| extdata | JSON for any extra data. |

**rocpd_arg**

Purpose: Stores arguments (name, type, value, position) for events; each row references an event via event_id. Commonly used for API parameters (e.g. HIP) and other event-specific details.

| Field | Meaning |
| ----- | ------- |
| id | Primary key. |
| guid | Database/import identifier. |
| event_id | References rocpd_event.id. |
| position | Argument position (order). |
| type | Argument type. |
| name | Argument name. |
| value | Argument value (text). |
| extdata | JSON for any extra data. |

**rocpd_pmc_event**

Purpose: Records a performance-counter value for an event; links a counter (pmc_id) to an event (event_id) and stores the numeric value.

| Field | Meaning |
| ----- | ------- |
| id | Primary key. |
| guid | Database/import identifier. |
| event_id | References rocpd_event.id (may be null). |
| pmc_id | References rocpd_info_pmc.id. |
| value | Counter value (numeric). |
| extdata | JSON for any extra data. |

**rocpd_region**

Purpose: CPU-side timed interval (start/end on one thread); stores name_id and event_id. Used for API calls and CPU regions in traces.

| Field | Meaning |
| ----- | ------- |
| id | Primary key. |
| guid | Database/import identifier. |
| nid | References rocpd_info_node.id. |
| pid | References rocpd_info_process.id. |
| tid | References rocpd_info_thread.id. |
| start | Region start timestamp (nanoseconds). |
| end | Region end timestamp (nanoseconds). |
| name_id | References rocpd_string.id for the region name. |
| event_id | References rocpd_event.id (may be null). |
| extdata | JSON for any extra data. |

**rocpd_sample**

Purpose: Single-timestamp (instant) sample on a track; references track_id and event_id. Used for sampling-based profiles.

| Field | Meaning |
| ----- | ------- |
| id | Primary key. |
| guid | Database/import identifier. |
| track_id | References rocpd_track.id. |
| timestamp | Sample timestamp (nanoseconds). |
| event_id | References rocpd_event.id (may be null). |
| extdata | JSON for any extra data. |

**rocpd_kernel_dispatch**

Purpose: One GPU kernel launch: agent, kernel, queue, stream, start/end timestamps, and grid/workgroup sizes. The main event table for GPU kernel execution.

| Field | Meaning |
| ----- | ------- |
| id | Primary key. |
| guid | Database/import identifier. |
| nid | References rocpd_info_node.id. |
| pid | References rocpd_info_process.id. |
| tid | References rocpd_info_thread.id (may be null). |
| agent_id | References rocpd_info_agent.id (GPU that executed the kernel). |
| kernel_id | References rocpd_info_kernel_symbol.id. |
| dispatch_id | Dispatch identifier. |
| queue_id | References rocpd_info_queue.id. |
| stream_id | References rocpd_info_stream.id. |
| start | Kernel start timestamp (nanoseconds). |
| end | Kernel end timestamp (nanoseconds). |
| private_segment_size | Scratch memory size for this dispatch. |
| group_segment_size | LDS size for this dispatch. |
| workgroup_size_x, workgroup_size_y, workgroup_size_z | Workgroup dimensions. |
| grid_size_x, grid_size_y, grid_size_z | Grid dimensions. |
| region_name_id | References rocpd_string.id for the API region name (may be null). |
| event_id | References rocpd_event.id (may be null). |
| extdata | JSON for any extra data. |

**rocpd_memory_copy**

Purpose: One memory copy operation (e.g. host-to-device, device-to-host); stores start/end, size, src/dst agent and address, and event_id.

| Field | Meaning |
| ----- | ------- |
| id | Primary key. |
| guid | Database/import identifier. |
| nid | References rocpd_info_node.id. |
| pid | References rocpd_info_process.id. |
| tid | References rocpd_info_thread.id (may be null). |
| start | Copy start timestamp (nanoseconds). |
| end | Copy end timestamp (nanoseconds). |
| name_id | References rocpd_string.id for the copy operation name. |
| dst_agent_id | References rocpd_info_agent.id (destination device). |
| dst_address | Destination address. |
| src_agent_id | References rocpd_info_agent.id (source device). |
| src_address | Source address. |
| size | Copy size in bytes. |
| queue_id | References rocpd_info_queue.id (may be null). |
| stream_id | References rocpd_info_stream.id (may be null). |
| region_name_id | References rocpd_string.id for the API region name (may be null). |
| event_id | References rocpd_event.id (may be null). |
| extdata | JSON for any extra data. |

**rocpd_memory_allocate**

Purpose: One memory allocation event (ALLOC, FREE, REALLOC, or RECLAIM) at REAL, VIRTUAL, or SCRATCH level; stores size, address, agent, and event_id.

| Field | Meaning |
| ----- | ------- |
| id | Primary key. |
| guid | Database/import identifier. |
| nid | References rocpd_info_node.id. |
| pid | References rocpd_info_process.id. |
| tid | References rocpd_info_thread.id (may be null). |
| agent_id | References rocpd_info_agent.id (may be null). |
| type | One of: 'ALLOC', 'FREE', 'REALLOC', 'RECLAIM'. |
| level | One of: 'REAL', 'VIRTUAL', 'SCRATCH'. |
| start | Operation start timestamp (nanoseconds). |
| end | Operation end timestamp (nanoseconds). |
| address | Allocated or freed address (may be null). |
| size | Size in bytes. |
| queue_id | References rocpd_info_queue.id (may be null). |
| stream_id | References rocpd_info_stream.id (may be null). |
| event_id | References rocpd_event.id (may be null). |
| extdata | JSON for any extra data. |
