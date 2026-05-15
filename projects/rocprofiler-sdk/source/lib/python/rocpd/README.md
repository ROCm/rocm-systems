# ROCm Profiling Data (RocPD)

The RocPD Python package provides a scriptable API for analyzing, summarizing, filtering, and merging tracing data
collected with the ROCm profiling tools suite.

## Background

In the past, the ROCm profiling tools (e.g. rocprofv3, rocprofiler-systems, etc.) have directly written data to
various output formats such as CSV, JSON, Perfetto, OTF2, etc. This approach has a significant number of flaws:

### No standardization in the CSV and JSON output formats

The ROCm profiling groups considers the standardization of the CSV and JSON output formats for all the tools
as a waste of time. Neither of these data formats scale well when large amounts of profiling data is collected
Due to the inherent overhead of parsing textual data as opposed to binary, the archane simplicity of the
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

In order to unify the output colleted across multiple processes, the one-output-per-process approach
requires either (A) a post-processing step which combines the various outputs into a single output,
(B) an output format which utilizes a single "metadata" file which links together the individual
outputs, or (C) a visualizer which supports opening multiple files at once. The ROCm profiling group
considers Option A are the most flexible and reliable approach since Option B does require a small
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
`rocpd.importer.RocpdImportData` object.

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
