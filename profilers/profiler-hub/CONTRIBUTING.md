# Contributing to profiler-hub #

Contributions to profiler-hub are welcome. Contributions must conform to the license identified in
[LICENSE.md](LICENSE.md) and pass the project's test requirements (`ctest`). The author must also be
able to respond to review comments and make any changes requested.

For contribution process shared across rocm-systems (sparse checkout, branching model, review
routing, large-file storage) see the [super-repo CONTRIBUTING.md](../../CONTRIBUTING.md).

## Issue Discussion ##

Please use the GitHub Issues tab to let us know of any issues.

* Use your best judgment when creating issues. If your issue is already listed, please upvote the
  issue and comment or post to provide additional details, such as the way to reproduce this issue.
* If you're unsure if your issue is the same, err on caution and file your issue. You can add a
  comment to include the issue number (and link) for a similar issue. If we evaluate your issue as
  being the same as the existing issue, we'll close the duplicate.
* If your issue doesn't exist, use the issue template to file a new issue.
  * When you file an issue, please provide as much information as possible, including relevant
    output, so we can get information about your configuration. This helps reduce the time required
    to reproduce your issue.
  * Check your issue regularly, as we may require additional information to reproduce the issue
    successfully.
* You may also open an issue to ask the maintainers whether a proposed change meets the acceptance
  criteria or to discuss an idea about the library.

## Acceptance Criteria ##

GitHub issues are recommended for any significant change to the code base that adds a feature or
fixes a non-trivial issue. If the code change is large without the presence of an issue (or prior
discussion with AMD), the change may not be reviewed. Small fixes that fix broken behavior or other
bugs are always welcome with or without an associated issue.

## Pull Request Guidelines ##

By creating a pull request, you agree to the statements made in the [Code License](#code-license)
section below.

1. Target the `develop` integration branch.
2. Associate the PR with one JIRA ticket, referenced in the title prefix and the body (e.g.
   `ROCPDSNA-NN: <summary>`).
3. Fill out the repository PR template completely; do not leave its boilerplate in place unedited.
4. Keep each PR reviewable by a single reviewer group. Split cross-subsystem or cross-repo work into
   separate, coordinated PRs rather than bundling unrelated changes.
5. Put rationale in the PR description, not in code comments: describe what changed, why, and how it
   was tested, with any guidance a reviewer needs. Keep inline code comments concise; they document
   the code, not the change history.
6. New or changed behavior should come with test coverage. A reviewer should treat a change that adds
   behavior without an accompanying test as incomplete.
7. Mechanical checks enforced in CI: `clang-format-18` and `gersemi` must produce no diff on changed
   C/C++ and CMake files. `clang-tidy` also runs in CI and its findings are reported on the PR, but as
   of this writing a `clang-tidy` finding does not by itself fail the job — treat new findings on
   touched code as something to fix before merge, not something CI will catch for you. Text files
   should end with a newline (LF, no CRLF). Compiler warnings introduced on touched library-target
   code should be addressed, even though the library targets are not built with `-Werror`.
8. A submodule pin must point at a landed upstream commit, never a personal or deletable branch.
9. Open PRs as draft; flip to ready for review after self-review and a green CI run.
10. Commit subjects follow `<scope>: <imperative verb phrase>` with `profiler-hub` as the scope (e.g.
    `profiler-hub: fix null storage pointer in writer_t ctor`); the body is a tight bullet list leading
    with *what* changed and then *why*, with backticks around filenames and identifiers.

These are proposed norms drawn from the team's own PR practice; the team ratifies or amends them over
time.

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
2. Public API headers live in [`include/`](include) at the project root; everything else lives under
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

## Testing Guidelines ##

1. Unit tests use GTest and live in [`tests/unit/`](tests/unit), one file per subject named
   `<subject>_test.cpp` (e.g. `writer_test.cpp` tests `writer.hpp`/`writer_impl.cpp`). Most files
   declare a single fixture named `<subject>_test : public ::testing::Test`; a file covering several
   related builders may instead declare multiple fixtures named for what each one exercises (e.g.
   `insert_query_builders_test.cpp` declares `query_value_builder_test` and
   `query_columns_builder_test`).
2. Performance tests use Google Benchmark and live in [`tests/benchmarks/`](tests/benchmarks), named
   `<subject>_bench.cpp`.
3. [`tests/find_package/`](tests/find_package) is a smoke test that consumes profiler-hub the way an
   external project would, via `find_package(profiler-hub REQUIRED)` — it guards the installed
   package/target interface, not internal behavior.
4. Unit tests and the find_package smoke test are wired up through `gtest_discover_tests`/`add_test`
   respectively, so `ctest` picks them up; there is no separate ad hoc test runner for them.
   Benchmarks are not registered with `ctest` — they are run directly as a standalone executable.

## Formatting & Tooling ##

Run these locally before committing; both are enforced in CI:

```bash
# Format changed C++ files (uses this repo's .clang-format, clang-format-18)
clang-format-18 -i <changed .hpp/.cpp files>

# Static analysis (uses this repo's .clang-tidy)
cmake --build build --target clang-tidy        # report only
cmake --build build --target clang-tidy-fix    # apply automatic fixes
```

## Code License ##

All code contributed to this project will be licensed under the license identified in
[LICENSE.md](LICENSE.md). Your contribution will be accepted under the same license.

## Release Cadence ##

Any code contribution to this library will be released with the next version of ROCm if the
contribution window for the upcoming release is still open. If the contribution window is closed but
the PR contains a critical security/bug fix, an exception may be made to include the change in the
next release.

## References ##

* [Super-repo CONTRIBUTING.md](../../CONTRIBUTING.md) — shared rocm-systems contribution workflow.
* [`.clang-format`](.clang-format) / [`.clang-tidy`](.clang-tidy) — mechanical style and lint source
  of truth.
* [`include/`](include) — the public API surface.
