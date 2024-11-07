# rocprofv3 Multi-Node Profiling Data

## Overview

- rocprofv3 adds supports for a `--output-format rocpd` option which enables writing a SQLite database file (one per process) with the collected data
  - Use SQL schema from `rocpd` initially to support the rocpd post-processing analysis support
- In order to visualize the data, users will convert the database(s) to their desired visualization formats
  - SQL has a relatively easy way to treat multiple separate databases as one database via views
- rocprofv3 provides some command-line tools built on top of a python package designed for post-processing our databases

### Skills Required for Tasks

1. Rework rocprofv3 tool library output functions
    - __C++__: output functions written in C++ (`^/source/lib/rocprofiler-sdk-tool/generate*`)
    - __CMake__: move the output functions into stand-alone library
2. Create Python package skeleton in `^/source/lib/python`
    - __Python__: organizing a Python package to be importable (`import rocpd`) and executable (i.e. `python -m rocpd --help`)
3. Adding rocprofv3 SQLite support
    - __C++__: just a general skill requirement for working with rocprofiler-sdk
    - __CMake__: for integrating SQLite and python bindings into rocprofiler-sdk build
    - __SQL__: understanding of SQL statement meanings, knowledge of `rocpd` SQL schema
4. Python bindings for output functions
    - __C++__: just a general skill requirement for working with rocprofiler-sdk
    - __PyBind11__: for writing Python bindings

#### Task #1: Rework `rocprofv3` Tool Library Output Functions

The problems with most of the output functions are:

- Problem: Access global memory via `tool_table` functions
  - Global memory access won't work well for invocation of these functions via Python bindings
    - Ideally, these functions should be written in the (pseudo-) functional programming style, i.e., function only accesses memory of arguments, communicates via return value, and avoids concepts like shared states but without restrictions such as immutable data arguments
- Problem: Require all the profiling data to be loaded into memory
  - During runtime, rocprofv3 writes data to buffer and when buffer is full, writes the binary blob to a temporary intermediate binary file
  - During finalization, rocprofv3 reads _all_ of this data back into memory from the intermediate binary file and then writes to various output forms
    - This approach will not work when amount of collected data exceeds amount of available RAM, especially on systems with swap disabled; e.g., 1 TB of profiling data on system with 128 GB of RAM
  - We need to be able to stream data in chunks to these output functions
    - Proposed approach: function which creates a file handle, function which writes chunk of data to file (invoked multiple times), function which closes file handle

> Assigned: Markus, Olha, Jin, Araceli (i.e. onboarding group task) + Jonathan (CMake part)

##### Tasks

1. Move the `source/lib/rocprofiler-sdk-tool/generate*.{hpp,cpp}` functions into standalone (static) library: `source/lib/tool-data`
    - May require `source/lib/tool-common` (static) library if something is needed by both `tool-data` and `rocprofiler-sdk-tool` libraries
    - Please consult if you have any questions about where to put things and/or naming conventions
    - Pay attention to existing CMake and use similar style
    - We will link this library into `rocprofiler-sdk-tool` and link it into Python bindings library
2. Solve global memory access problem
   - Probably need some additional data structures which represents the data currently stored/accessed from global memory which will be passed into function.

### Python Package for Converting Databases to Other Output Formats

> __Note__: We could potentially reuse `rocpd` for the the python package name since "ROCm Profiling Data" is a pretty appropriate name.

rocprofv3 will need to rework the output functions within the `librocprofiler-sdk-tool.so` library (underlying library used by `rocprofv3`) in order to support Python bindings.
For example, `generateJSON(...)` currently fetches info from global memory stored during the run, we need these functions to be pure: the only memory operated on is from the function arguments.
Furthermore, these output functions need to support partial writes: invocations with only a subset of the data so that all the data need not be loaded into memory at one time.

> __Example__ (workflow): get handle to output format, e.g. a Perfetto session, invoke `generatePerfetto(...)` with some of the data, repeat until all data has been passed, close handle to the output format.

These reworked functions should be moved to another library, e.g. `librocprofiler-sdk-tool-io.(a|so)`.
Once the output functions are isolated and functional, we need to generate python bindings (via PyBind11) so that a python package can be built on top of them.
Various command-line tools can be provided using `__main__.py` file(s) within our python package.
Users can use the python package to write their own scripts.

> __Example__ (two databases, one Perfetto trace): `rocprofv3-merge --output-format pftrace --out mybenchmark.pftrace --in results-1000.db results-1001.db`

### Treating multiple SQL databases as one database

```python
conn = sqlite3.connect('db1.db')
conn.execute("ATTACH DATABASE 'db2.db' AS db2;")
conn.execute("ATTACH DATABASE 'db3.db' AS db3;")

# Create a view that unifies the 'users' table from all three databases
conn.execute("""
CREATE VIEW all_users AS
SELECT * FROM users
UNION ALL
SELECT * FROM db2.users
UNION ALL
SELECT * FROM db3.users;
""")

# Now you can query the view as if it were a single table
cursor = conn.execute("SELECT * FROM all_users;")
for row in cursor:
    print(row)

# Close the connection
conn.close()
```

## Proposed SQL Schema

A more comprehensive SQL Schema is proposed below. This schema is intended to be more comprehensive with respect to the
various types of data that profilers can collect (such as Omnitrace/RSP)

The schema consists of multiple interrelated tables to capture different categories of profiling data.
Below is a high-level schema with the primary tables and relationships.

__*Please note, this is a very preliminary sketch of the schema*__.
If you want to weigh in, please restrict comments to the high-level organization, comments that it doesn't contain
fields for correlation IDs or something like that are not particularly helpful at the moment.

```sql
CREATE TABLE strings (
    id SERIAL PRIMARY KEY,
    value VARCHAR(1024) UNIQUE
);

CREATE TABLE process (
    id INT PRIMARY KEY,
    pid INT,
    process_name_id INT,
    executable_path_id INT,
    start_time BIGINT,
    end_time BIGINT,
    FOREIGN KEY (process_name_id) REFERENCES strings(id)
    FOREIGN KEY (executable_path_id) REFERENCES strings(id)
);

CREATE TABLE thread (
    id INT PRIMARY KEY,
    tid INT,
    process_id INT,
    thread_name_id INT,
    start_time BIGINT,
    end_time BIGINT,
    FOREIGN KEY (process_id) REFERENCES process(process_id),
    FOREIGN KEY (thread_name_id) REFERENCES strings(id)
);

CREATE TABLE cpu_info (
    id SERIAL PRIMARY KEY,
    core_id INT,
    socket_id INT,
    frequency_hz INT,
    model_id INT,
    cache_size_kb INT,
    FOREIGN KEY (model_id) REFERENCES strings(id)
);

CREATE TABLE gpu_info (
    id SERIAL PRIMARY KEY,
    device_name_id INT,
    compute_capability_id INT,
    memory_size_mb INT,
    multiprocessor_count INT,
    clock_rate_hz INT,
    FOREIGN KEY (device_name_id) REFERENCES strings(id)
    FOREIGN KEY (compute_capability_id) REFERENCES strings(id)
);

CREATE TABLE instrumentation_regions (
    id SERIAL PRIMARY KEY,
    process_id INT,
    thread_id INT,
    region_name_id INT,
    start_time BIGINT,
    end_time BIGINT,
    parent_region_id INT,
    duration_ns BIGINT GENERATED ALWAYS AS (end_time - start_time) STORED,
    file_id INT,
    line_number INT,
    additional_info JSONB,
    FOREIGN KEY (process_id) REFERENCES process(process_id),
    FOREIGN KEY (thread_id) REFERENCES thread(thread_id),
    FOREIGN KEY (region_name_id) REFERENCES strings(id),
    FOREIGN KEY (file_id) REFERENCES strings(id)
);

CREATE TABLE call_stacks (
    id SERIAL PRIMARY KEY,
    process_id INT,
    thread_id INT,
    timestamp BIGINT,
    stack_depth INT,
    function_id INT,
    file_id INT,
    line_number INT,
    parent_sample_id INT,
    call_site VARCHAR(1024),
    additional_info JSONB,
    FOREIGN KEY (process_id) REFERENCES process(process_id),
    FOREIGN KEY (thread_id) REFERENCES thread(thread_id),
    FOREIGN KEY (function_id) REFERENCES strings(id),
    FOREIGN KEY (file_id) REFERENCES strings(id)
);

CREATE TABLE hardware_counters (
    id SERIAL PRIMARY KEY,
    process_id INT,
    thread_id INT,
    timestamp BIGINT,
    event_id INT,
    value BIGINT,
    FOREIGN KEY (process_id) REFERENCES process(process_id),
    FOREIGN KEY (thread_id) REFERENCES thread(thread_id),
    FOREIGN KEY (event_id) REFERENCES strings(id)
);

CREATE TABLE memory_operations (
    id SERIAL PRIMARY KEY,
    process_id INT,
    thread_id INT,
    timestamp BIGINT,
    operation_type VARCHAR(50) CHECK (operation_type IN ('ALLOC', 'FREE', 'COPY')),
    source_address BIGINT,
    destination_address BIGINT,
    size_bytes BIGINT,
    duration_us BIGINT,
    additional_info JSONB,
    FOREIGN KEY (process_id) REFERENCES process(process_id),
    FOREIGN KEY (thread_id) REFERENCES thread(thread_id)
);

CREATE TABLE gpu_kernel_launches (
    id SERIAL PRIMARY KEY,
    process_id INT,
    thread_id INT,
    gpu_id INT,
    kernel_id INT,
    dispatch_id INT,
    launch_time BIGINT,
    start_time BIGINT,
    end_time BIGINT,
    grid_size_x INT,
    grid_size_y INT,
    grid_size_z INT,
    block_size_x INT,
    block_size_y INT,
    block_size_z INT,
    shared_mem_bytes INT,
    duration_ns BIGINT GENERATED ALWAYS AS (end_time - start_time) STORED,
    FOREIGN KEY (process_id) REFERENCES process(process_id),
    FOREIGN KEY (thread_id) REFERENCES thread(thread_id),
    FOREIGN KEY (gpu_id) REFERENCES gpu_info(gpu_id),
    FOREIGN KEY (kernel_id) REFERENCES strings(id)
);

CREATE TABLE binary_analysis_info (
    id SERIAL PRIMARY KEY,
    process_id INT,
    binary_name VARCHAR(1024),
    function_id INT,
    start_address BIGINT,
    end_address BIGINT,
    instruction_count INT,
    file_id INT,
    line_number INT,
    call_sites JSONB,
    additional_info JSONB,
    FOREIGN KEY (process_id) REFERENCES process(process_id),
    FOREIGN KEY (function_id) REFERENCES strings(id),
    FOREIGN KEY (file_id) REFERENCES strings(id)
);
```

Explanation of the design considerations:

1. __Separate String Tables__: Created unique string tables for function names, file names, kernel names, and event names to avoid storing redundant copies.
    - `function_names`, `file_names`, `kernel_names`, and `event_names` tables are created to hold unique strings. Each table has a surrogate primary key (`function_id`, `file_id`, `kernel_id`, `event_id`) that is referenced by the main tables.
    - This avoids storing redundant copies of long or frequently repeating strings in different tables, reducing the storage footprint and improving consistency.
2. __Foreign Key References__: Main tables reference unique strings using foreign keys for consistency and space efficiency.
    - Main tables such as `instrumentation_regions`, `call_stacks`, `gpu_kernel_launches`, etc., reference these unique string tables using foreign keys.
    - This makes querying for specific function names or kernel names more efficient, as the strings are indexed separately.
3. __Computed Columns__: Used computed columns for duration fields to facilitate quick analysis.
    - The `duration_us` columns are computed based on timestamps, providing useful metrics for quick analysis.
4. __Extensibility__: Designed to be easily extensible with additional string categories if needed.
    - New string types or categories can be added by creating new tables, and the main tables can reference them with minor schema adjustments.
5. __JSONB for Additional Metadata__:
    - JSONB columns (`additional_info`) are used to handle complex or variable metadata that doesn’t fit neatly into the structured schema (e.g., custom annotations, extra debug info).

### Example Data Insertion and Lookup

#### Adding a new function

```sql
INSERT INTO function_names (function_name) VALUES ('my_function') ON CONFLICT (function_name) DO NOTHING;
```

#### Linking a function in a call stack

```sql
INSERT INTO call_stacks (process_id, thread_id, timestamp, stack_depth, function_id, file_id)
VALUES (123, 456, '2024-09-27 10:00:00', 1, (SELECT function_id FROM function_names WHERE function_name = 'my_function'),
        (SELECT file_id FROM file_names WHERE file_name = 'my_file.c'));
```

## Q & A

### All global variables are protected with locks in common synchronized library. How are we sending the data from these variables to the pure functions?

There is a new `rocprofiler::tool::metadata` struct in `lib/output/metadata.hpp` which will be populated with data from SQL.
This struct is passed to the output functions.

### If we provide the functionality to flush the trace at regular intervals, do we delete the data in global memory after each flush? If not, how do we keep track of data already read at any given point time during runtime?

We will probably not delete the metadata (agent info, code objects, kernel symbols, etc.) after a flush.
When we flush, we will swap out the temporary binary file with a new temporary binary file and write/append the database with
the contents of the old temporary binary file.

### Can a user collect trace at regular flush interval and ask for counter collection at the end of application?

I am not sure what you mean here. We can write counter collection data when we flush. If the user is asking for periodic
flushing, we will restrict the output format to the database. In other words, I suspect that only `--flush-rate X` will only
be compatible with `--output-format rocpd` -- any additional or alternative data formats and we will throw an error in the
rocprofv3 script. This is for simplicity sake, supporting periodically flushing to CSV, etc. is unnecessary in my opinion.

### I think hardware_counters table in database schema should have a dispatch_id field to represent the kernel it belongs to

Please note, the proposed schema states clearly:

> __*Please note, this is a very preliminary sketch of the schema*__.
> If you want to weigh in, please restrict comments to the high-level organization, comments that it doesn't contain
> fields for correlation IDs or something like that are not particularly helpful at the moment.

However, I will note that the hardware counters table is probably going to be generic, i.e. supporting CPU HW counters, which
do not have dispatch IDs. Lastly, I will also note, device counter collection is not associated with a dispatch so even in
the case of GPU HW counters, including this field is questionable.

### What is binary analysis info table?

More advanced tools such as Omnitrace/Rocprofiler-System do address to line translations. This could also potentally
include the sort of data related to PC sampling

### What is the Key of gpu info table? Node_id/zero based numbering scheme?

That isn't defined. Very preliminary sketch.

### When is user allowed to access the database in case of flushing the trace at regular intervals? Is user allowed to read the database only after tool finalization? Or we create a database file for each interval?

TBD on the exact details but the user will certainly be able to read the database before tool finalization when it is flushed.
