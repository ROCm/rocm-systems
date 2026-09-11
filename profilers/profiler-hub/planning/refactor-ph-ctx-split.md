# Refactor: Split ph_ctx class from C ABI glue

## Goal
`source/profiler_hub_ctx.hpp` currently defines the `ph_ctx` class fully
inline (ctor + all methods in-class). `source/profiler_hub_ctx.cpp` does not
implement the class at all — it only implements the `extern "C"` glue
functions declared in `profiler_hub.h`. Names suggest the opposite (cpp =
class impl). Split so:
- `profiler_hub_ctx.hpp` / `profiler_hub_ctx.cpp` = `ph_ctx` class
  declaration + out-of-line method implementations only.
- New `profiler_hub_c_api.cpp` = the `extern "C"` glue functions
  (`ph_ctx_create`, `ph_ctx_free`, `ph_get_library_version`,
  `ph_get_schema_version`, `ph_get_track_list`, `ph_get_node`).

## Code Smells Identified
- [x] Misleading file/impl split: `.cpp` doesn't implement what `.hpp`
      declares; C API glue is unrelated to `ph_ctx`'s own implementation.
- [x] Header contains implementation (all `ph_ctx` methods inline) though
      not template code — no reason to keep bodies in the header.

## Refactoring Strategy

### Best Practices to Apply
- Standard header/impl split: declarations in `.hpp`, definitions in `.cpp`.
- No behavior change; pure move of code between files.

### Design Patterns to Use
No patterns needed.

### STL Algorithms to Apply
No algorithm replacements (out of scope for this move).

## Tasks
- [x] Rewrite `profiler_hub_ctx.hpp`: class declaration only (ctor + method
      signatures + members), drop bodies
- [x] Rewrite `profiler_hub_ctx.cpp`: out-of-line `ph_ctx::` method
      definitions (ctor, get_storage_version, get_track_list, get_node,
      initialize_c_track_list, initilaize_node_info)
- [x] Create `source/profiler_hub_c_api.cpp` with the `extern "C"` glue
      functions moved out of `profiler_hub_ctx.cpp`
- [x] Update `source/CMakeLists.txt` to add the new source file
- [x] Build project, confirm no new warnings/errors

## Testability Improvements
Already testable at same level as before (thin ABI glue vs. class body); no
change in testability, just file organization.

## Changelog
Not added to changelog (internal file reorganization, no behavior/API change).

## Notes
Keep public API (profiler_hub.h) unchanged. No signature changes.
