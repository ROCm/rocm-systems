# Documentation: c_interface headers

## Goal
Add Doxygen documentation to public C ABI headers in `include/c_interface/`.

## Type
API Documentation (Code Comments)

## Audience
External C/C++ consumers of `libprofiler-hub` linking against the C ABI. Need
ownership/lifetime rules, error codes, and ABI stability notes since this is
a boundary they can't just "read the .cpp" to understand.

## Analysis
- `profiler_hub_types.h`: types/enums, no docs at all.
- `profiler_hub.h`: function declarations, no docs at all.
- Known open issues from review (memory only, not fixed here):
  - `ph_track_t.track_name` and other `const char*` fields point into
    ctx-owned storage; must document lifetime tied to `ph_ctx_t`.
  - `ph_track_list_t.tracks` pointer same lifetime rule.
  - `ph_future_*` declared, unimplemented — document as not yet implemented.
  - `ph_get_node` currently buggy (output not populated) — doc describes
    intended contract, not current buggy behavior; do not paper over the bug.

## Tasks
- [x] Document `profiler_hub_types.h`: enum, typedefs, structs, field lifetimes
- [x] Document `profiler_hub.h`: all function declarations (params, return,
      ownership, error conditions)
- [x] Note `ph_future_*` as unimplemented in current version

## Structure
- File-level `@file` brief for both headers
- `@brief` + `@param`/`@return` per function
- `@note` for lifetime/ownership caveats on pointer-bearing structs/fields
