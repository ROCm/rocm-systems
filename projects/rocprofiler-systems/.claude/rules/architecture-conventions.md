# Architecture Conventions — rocprofiler-systems

Use this when adding or changing C++ under `source/lib/`. Paths are relative to
`projects/rocprofiler-systems/`.

## Reuse before writing

Before writing a helper, search these locations and use what exists:

| Need | Location |
| --- | --- |
| Units: data size, frequency, power, generic quantity | `source/lib/common/units/` |
| String split/join/trim, delimiting | `source/lib/common/string_utility.hpp`, `delimit.hpp`, `join.hpp` |
| Paths, environment, env vars | `source/lib/common/path.hpp`, `environment.hpp`, `env_vars.hpp` |
| Version handling | `source/lib/common/version.hpp` |
| Logging | `LOG_*` macros from `source/lib/logger/debug.hpp` |
| Concepts for compile-time DI | `source/lib/policies/` |
| External library access | `source/lib/backends/<library>/` |

If no helper fits, add the generic part to `source/lib/common/` and keep only
the feature-specific code in your module. Log with `LOG_CRITICAL`, `LOG_ERROR`,
`LOG_WARNING`, `LOG_INFO`, `LOG_DEBUG` or `LOG_TRACE`. Do not use `printf`,
`std::cout`, or timemory output.

## External dependencies: wrapper + backend

Code outside `source/lib/backends/` never calls an external library directly
(rocprofiler-sdk, amd-smi, hipFile, ...). Every library gets a directory
`source/lib/backends/<library>/` with these files:

1. `wrapper.hpp` — a 1:1 seam. Each static method makes one library call and
   does nothing else: no error interpretation, no state, no logic.
   - Strip the library prefix from names: `amdsmi_shut_down()` becomes
     `wrapper::shutdown()`, and `hipFileStatsLevel3_t` becomes `wrapper::stats_l3_t`.
   - This is the only file that includes the library's public header.
2. `backend.hpp` — `template <typename Wrapper> class backend`. It calls only
   `Wrapper::` and holds the logic: error checks, conversions, caching.
   Production code instantiates it with the real wrapper. Tests instantiate it
   with a mock wrapper.
3. `types.hpp` (optional) — project-owned types that the backend exposes, so
   callers never see library types.

Keep the shape and call order the library already has. Do not invent a new
abstraction over it. Copy the structure of the closest existing backend:
`hipfile/` (small, dlsym-loaded), `amd_smi/` (handle-based), or
`rocprofiler_sdk/` (large, callback-based).

## Concepts in `source/lib/policies/`

- A concept header includes only standard headers and other headers from
  `policies/`. It never includes a backend, a library header, or `core/`.
- A concept can compose other concepts. For example, `agent_manager_policy`
  builds on `agent_policy`.
- Write each concept to state the minimum its consumer needs, so any conforming
  type, including test fakes, can satisfy it.

## Trace data: `source/lib/core/trace_cache/`

Collectors record trace data at runtime. It is converted to output formats only
after the run finishes:

1. **Record (hot path).** `trace_cache::get_buffer_storage().store(sample)`
   takes one mutex and `memcpy`s a type header and the serialized sample into a
   100 MB in-memory ring buffer. A background thread writes the buffer to
   `/tmp/buffered_storage_<ppid>_<pid>.bin` once 80 MB are used.
2. **Metadata (once per entity).** `trace_cache::get_metadata_registry().add_*()`
   records threads, tracks, PMC info, queues, streams, strings and code objects
   in deduplicated sets. These are saved to `/tmp/metadata_<ppid>_<pid>.json`.
3. **Post-process (at finalization, root process only).**
   `cache_manager::post_process_bulk()` finds every cache file. `storage_parser`
   reads each record, and `type_registry` turns it back into its sample type.
   The sample is then passed to `sample_processor_interface::handle()` on each
   output processor: `perfetto_processor`, `rocpd_processor`.

Keep `store()` fast:

- A sample holds raw values: timestamps, IDs, handles, enums. Look up names,
  build strings and format output in post-processing, not before `store()`.
- Data that is the same for every sample (thread info, track name, PMC
  description) goes into the metadata registry once. The sample stores only
  the ID that refers to it.
- Do not wrap `store()` in your own lock. `store()` throws if the storage is
  not running.
- `std::string_view` fields are copied into the buffer, so the viewed data only
  has to live until `store()` returns. In a parsed sample, a `string_view`
  points into the parser's read buffer, which is reused for the next record.
  A processor copies it if it keeps it after `handle()` returns.

To add a sample type, copy an existing one:

1. Add a new value to `type_identifier_t`. Never renumber or reuse a value.
   `fragmented_space` is reserved.
2. Write a struct that derives from `cacheable_t` and declares
   `static constexpr type_identifier_t type_identifier`. Allowed field types:
   trivially copyable values, `std::string_view`, `std::vector` or `std::span`
   of trivially copyable values, and `std::optional` (not nested).
3. Specialize `serialize`, `deserialize` and `get_size`. All three list the same
   fields in the same order.
4. Add the type to `storage_parser_t` in `storage_parser_alias.hpp`. Add a
   `handle()` overload to `sample_processor_interface`, and implement it in
   every processor.

## Unreachable cases fail loudly

When a `switch` over an enum handles every current value, add a `default:` that
throws `std::logic_error`. The message names the enum and says the value is not
handled. This turns a newly added enum value into an immediate, visible failure
instead of a silent wrong result. Nothing should catch this exception.

```cpp
switch(kind)
{
    case agent_kind::cpu: return handle_cpu();
    case agent_kind::gpu: return handle_gpu();
    default:
        throw std::logic_error(
            fmt::format("agent_kind {} not handled", static_cast<int>(kind)));
}
```

The same applies to any branch that the code's invariants say is impossible.

## Documentation

Write Doxygen (`/** @brief ... */` or `///`) on public API declared in headers:
types, public methods, and free functions other modules call. Private members,
`detail` namespaces, and `.cpp`-local helpers get a plain `//` comment only when
the *why* is not obvious from the code.
