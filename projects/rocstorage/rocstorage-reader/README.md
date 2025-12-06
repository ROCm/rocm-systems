# rocstorage-reader

A standalone library for reading ROCm profiler trace data from SQLite databases. This library provides data model classes and database reading functionality originally extracted from roc-optiq.

## Overview

rocstorage-reader provides:

- **Data Model Classes**: Core data structures for representing profiler trace data including Trace, Track, Slice, FlowTrace, StackTrace, ExtData, and Table objects.
- **Database Readers**: Support for reading ROCpd and ROCprof SQLite database formats with automatic format detection.
- **C Interface**: C-compatible API for use with Python/FFI bindings.
- **Async Operations**: Future-based asynchronous database reading operations.

## Building

### As part of rocstorage

The recommended way to build is as part of the rocstorage project:

```bash
cd rocm-systems/projects/rocstorage
mkdir build && cd build
cmake .. -DROCSTORAGE_BUILD_READER=ON
make
```

### Standalone

rocstorage-reader can also be built standalone if sqlite3 and spdlog are available:

```bash
cd rocstorage-reader
mkdir build && cd build
cmake ..
make
```

## Running Tests

Unit tests are available when building with tests enabled:

```bash
cmake .. -DROCSTORAGE_READER_BUILD_TESTS=ON
make rocstorage_reader_unit_tests
ctest -L unit
```

## Using with roc-optiq

roc-optiq can use either its internal datamodel or the external rocstorage-reader library:

```bash
cd roc-optiq
mkdir build && cd build
cmake .. \
    -DUSE_EXTERNAL_ROCSTORAGE_READER=ON \
    -DROCSTORAGE_READER_PATH=/path/to/rocstorage/build/rocstorage-reader
make
```

When running, ensure the library is accessible:

```bash
export LD_LIBRARY_PATH=/path/to/rocstorage/build/rocstorage-reader:$LD_LIBRARY_PATH
./roc-optiq
```

## API Overview

### C Interface

```c
#include "rocprofvis_c_interface.h"

// Create a trace object
rocprofvis_dm_trace_t trace = rocprofvis_dm_create_trace();

// Open a database
rocprofvis_dm_database_t db = rocprofvis_db_open_database("trace.rpd", kAutodetect);

// Bind trace to database
rocprofvis_dm_bind_trace_to_database(trace, db);

// Create a future for async operations
rocprofvis_db_future_t future = rocprofvis_db_future_alloc(NULL, NULL);

// Read metadata asynchronously
rocprofvis_db_read_metadata_async(db, future);
rocprofvis_db_future_wait(future, 30);

// Get trace properties
uint64_t num_tracks = rocprofvis_dm_get_property_as_uint64(
    trace, kRPVDMNumberOfTracksUInt64, 0);

// Clean up
rocprofvis_db_future_free(future);
rocprofvis_dm_delete_trace(trace);
```

### C++ Interface

```cpp
#include "rocprofvis_dm_trace.h"
#include "rocprofvis_db.h"

// Create trace
auto trace = new RocProfVis::DataModel::Trace();

// Open database with auto-detection
auto db_type = RocProfVis::DataModel::Database::Autodetect("trace.rpd");
// Create appropriate database instance based on type...

// Bind and read data...

// Clean up
delete trace;
```

## License

MIT License - Copyright (c) 2025 Advanced Micro Devices, Inc.
