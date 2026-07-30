# profiler-hub Style Guide #

Conventions for writing C++ and CMake in the profiler-hub codebase. These describe the standards the
existing code already follows — prefer clarity and consistency with the surrounding code. For the
contribution process (issues, pull requests, testing, and licensing) see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Coding Style Guidelines ##

### Enforced vs. Judgment ###

Mechanical rules (indentation, brace placement, spacing, naming *enforcement*, include sorting,
cognitive-complexity limits) are owned by [`.clang-format`](.clang-format) and
[`.clang-tidy`](.clang-tidy) and are not restated here — those files are the source of truth. The
rest of this section covers the conventions that formatting and linting cannot check: how to
structure files, when to comment, how to shape a public API, and how tests are organized.

### File & Directory Conventions ###

1. Use `.hpp` for C++ headers and `.cpp` for C++ implementation files. profiler-hub has no `.h`/`.cc`
   files; there is no C-compatible API to distinguish.
2. Public API headers live in [`include/`](include) at the project root; the library implementation lives under
   [`source/`](source), organized by subsystem (`source/common`, `source/data_storage`,
   `source/queries`, `source/writers`). A subsystem with its own `CMakeLists.txt` (e.g.
   `source/queries`, `source/data_storage`) owns its own sources — never add another directory's
   files to your target; add a subdirectory and link its library/object target instead.
3. Public API types (`reader_t`, `writer_t`, `storage_t`) use the pimpl idiom: the header declares an
   opaque `struct impl;` and holds `std::unique_ptr<impl> m_impl;`, with the implementation confined
   to the corresponding `_impl.cpp`/`_impl.hpp`. This keeps the public headers free of internal
   dependencies (SQLite, spdlog, query builders) and stable across ABI-sensitive rebuilds.
4. Public API types are non-copyable and non-movable: copy constructor/assignment and move
   constructor/assignment are all `= delete`d, and the constructor is marked `explicit`. `reader_t`
   and `writer_t` follow this as a `struct` and additionally delete the default constructor
   explicitly (`reader_t() = delete;` / `writer_t() = delete;`). `storage_t` follows the same
   non-copyable/non-movable shape but is declared as a `class` with public:/private: sections, and
   its default constructor is suppressed implicitly — by the user-declared two-argument constructor
   — rather than deleted explicitly.

### Naming ###

`.clang-tidy` enforces `lower_case` for classes, structs, functions, variables, parameters,
constants, enum constants, and namespaces — including class/struct names. This matches the **C++
standard-library naming style** (snake_case throughout, including type names), not the Google C++
Style Guide (which pairs snake_case variables with PascalCase classes). profiler-hub is a low-level
storage library with a strong C/stdlib heritage, so the standard-library convention is the natural
fit and is the one already in force — use it, rather than PascalCase, for any new type.

Two suffix/prefix conventions are enforced by `.clang-tidy` and worth calling out explicitly since
they shape how you name new symbols:

- Type aliases and typedefs take a `_t` suffix (`reader_t`, `node_info_t`, `event_filter_t`).
- Private and protected member variables take an `m_` prefix (`m_impl`, `m_storage`, `m_backend`).

### Comments & Documentation ###

1. Public declarations in `include/` are documented with a Doxygen-style `/** @brief ... */` comment
   (with `@param`/`@return`/`@note` as applicable) since `include/` is the contract external callers
   rely on; this is the convention to strive for, and `reader.hpp`/`writer.hpp` follow it throughout.
   `storage.hpp` is the one present gap: its constructor and `get_storage_version()` currently carry
   no comment.
2. Implementation files (`source/*.cpp`, `*_impl.cpp`) are comment-sparse by comparison: no Doxygen
   blocks, and inline comments are reserved for non-obvious "why" — a subtle invariant, a workaround,
   a note on an easily-misread line. Do not carry header-style documentation into implementation
   files; the header already documents the contract.
3. Related groups of accessors in a public header may be introduced with a `@section` comment block
   (see `include/reader.hpp`) to separate, for example, cached accessors from on-demand query
   methods. Use this when a header has enough distinct accessor groups that a bare list of
   declarations would be hard to scan — not for every header.

### Error Handling ###

profiler-hub reports invalid input and broken invariants by throwing `std::invalid_argument` (bad
arguments — e.g. a null storage pointer) or `std::runtime_error` (failures discovered at runtime —
e.g. an unregistered entity). There is no error-code return convention in this codebase. Exception
messages state plainly what invariant was violated (built with `fmt::format` when interpolating
values), so a caller reading the message alone can tell what went wrong.

### CMake Conventions ###

1. Use target-based CMake exclusively: `target_sources`, `target_include_directories`,
   `target_compile_definitions`, `target_compile_options`, `target_link_libraries`. Directory-scoped
   commands like bare `include_directories()` or `add_definitions()` are not used anywhere in this
   codebase and should not be introduced.
2. Each subsystem CMakeLists (`source/queries/CMakeLists.txt`, `source/data_storage/CMakeLists.txt`)
   declares its own sources via `target_sources` and links into the shared `profiler-hub-objects`
   object library or its own static library; it does not reach across directories to add another
   subsystem's files.
3. The library targets (`profiler-hub-objects`, `profiler-hub_queries`) apply the same warning set:
   `-Wall -Wextra -Wshadow -Wvla -Wpedantic -Wconversion -Wsign-conversion -Wnon-virtual-dtor
   -Woverloaded-virtual -Wnull-dereference`. Add this set to any new library target; the test and
   benchmark targets do not currently carry it.

## Formatting & Tooling ##

Run these locally before committing; clang-format and gersemi are enforced in CI:

```bash
# Format changed C++ files (uses this repo's .clang-format, clang-format-18)
clang-format-18 -i <changed .hpp/.cpp files>

# Format changed CMake files (gersemi 0.25.1)
gersemi -i <changed CMakeLists.txt / *.cmake files>

# Static analysis (uses this repo's .clang-tidy)
cmake --build build --target clang-tidy        # report only
cmake --build build --target clang-tidy-fix    # apply automatic fixes
```
