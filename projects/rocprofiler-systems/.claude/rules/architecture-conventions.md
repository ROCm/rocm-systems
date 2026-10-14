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
