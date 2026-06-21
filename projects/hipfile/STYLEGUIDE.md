# hipFile Style Guide

This guide documents the coding conventions used in hipFile. Where a rule
is mechanically enforced (clang-format, ShellCheck, black, pylint), the
tool is authoritative; the prose here explains intent and covers what the
tools cannot check.

## File and Directory Structure

### Organization

* Platform-specific implementations live in `*_detail` directories. The
  AMD/HIP backend is in `amd_detail`; the NVIDIA backend is in
  `nvidia_detail` and is unmaintained.
* Code shared across backends lives in `common`.
* The public C API lives in `include`; private implementation headers sit
  next to their `.cpp` files in `src`.
* Tests mirror the source layout under `test` (e.g. `test/amd_detail`
  mirrors `src/amd_detail`).

### Naming

* Use hyphens, not underscores, in file and directory names
  (`file-descriptor.cpp`, not `file_descriptor.cpp`).
* `_detail` directories are the one exception and keep their underscore.
* A class's primary header/source pair is usually named after the concept
  it implements (`buffer.h`/`buffer.cpp`, `mountinfo.h`).

## Formatting

* All C and C++ code is formatted with **clang-format 18** using the
  project `.clang-format`. Run `util/format-source.sh` (it accepts an
  optional clang major version, e.g. `util/format-source.sh 18`).
* Key settings (see `.clang-format` for the full list):
    * 4-space indentation, no tabs
    * 110-column limit
    * Stroustrup brace style: opening brace on the same line, but `else`
      and `catch` go on their own line
    * Return type on its own line for top-level function *definitions*
    * Pointers bind to the name (`int *p`, not `int* p`)
    * Consecutive assignments, declarations, and macros are aligned
    * Includes are **not** auto-sorted (`SortIncludes: false`); order them
      by hand (see Header Guidelines)
    * Namespace comments are not added; inner namespaces are indented

## Naming Conventions

### C++

* **Types** (classes, structs, enums, type aliases): `PascalCase`
  (`Buffer`, `FilesystemType`, `UnregisteredFile`).
* **Interfaces** are abstract base classes prefixed with `I`
  (`IBuffer`, `IFile`, `IStream`).
* **Methods and member functions**: `camelCase` (`getBuffer`,
  `registerFile`, `isRegularFile`).
* **Free functions and function templates**: `snake_case`
  (`get_variant_ptr`, `is_variant_of_T_and_T_ptr`).
* **Enumerators** (in `enum class`): `snake_case` (`ext4`, `ordered`).
* **Local variables and parameters**: `snake_case`.
* **Namespace**: all library code lives in `namespace hipFile`.

### Member variable prefix

* Newer code prefixes private data members with `m_`
  (`m_client_fd`, `m_dio_mem_align`). Prefer this style for new members.
* Some older classes use bare `snake_case` members without the prefix.
  See `SUGGESTIONS_FOR_CONSISTENCY.md`.

### Public C API

* All public functions are prefixed `hipFile` and use `PascalCase` after
  the prefix (`hipFileBufRegister`, `hipFileDriverOpen`).
* Public macros and the error-code base are `UPPER_SNAKE_CASE` with a
  `HIPFILE_` prefix (`HIPFILE_VERSION_MAJOR`, `HIPFILE_BASE_ERR`).
* Error enumerators are `hipFile`-prefixed `PascalCase`
  (`hipFileDriverNotInitialized`), defined relative to `HIPFILE_BASE_ERR`.
* Platform-independent typedefs use a short lowercase name with a `_t`
  suffix (`hoff_t`, `hipFileHandle_t`).

### Python

* Modules: `snake_case` (and hyphen-free, per Python rules).
* Classes: `PascalCase` (`FileHandle`, `HipFileException`).
* Functions, methods, and variables: `snake_case`.
* "Private" attributes use a single leading underscore (`self._fd`).
* Class-level constants: `UPPER_SNAKE_CASE` (`DEFAULT_MODE`).
* Enums mirror the C API names verbatim so the binding stays recognizable.

## Comments and Documentation

* Every file starts with the standard copyright header and
  `SPDX-License-Identifier: MIT`.
* C++ symbols are documented with Doxygen using `///` line comments and
  `@brief`, `@param`, `@return`, `@attention` tags.
* The public C header uses Doxygen block comments (`/*! ... */` and
  `/** ... */`). Every documented symbol carries an `@ingroup` so it
  groups correctly in the generated docs.
* Concurrency requirements are documented on methods with `@attention`
  (e.g. which lock must be held).
* Python uses NumPy-style docstrings (a summary line, then
  `Parameters`/`Returns` sections).

## Header Guidelines

* Headers should follow [include-what-you-use](https://include-what-you-use.org/) guidelines.
* Use `#pragma once` instead of include guards.
* Place local headers (quoted) ahead of system headers (angle brackets),
  with each block in alphabetical order. clang-format does not sort
  includes, so this is maintained by hand.
* Public headers MUST have Doxygen markup for all public symbols.
* Private headers should have Doxygen markup flagged with `@internal`.
* Public headers MUST remain valid **C11** and contain no C++-isms; guard
  the `extern "C"` block with `#ifdef __cplusplus` and keep Windows
  portability shims behind `#ifdef _WIN32`.

## Language Features and Idioms

### CMake

* Requires CMake **3.21** (`cmake_minimum_required`); we have not moved
  to 4.x.
* Use modern (target-based) CMake paradigms and avoid "legacy CMake".
* `cmakelint` is configured via `.cmakelintrc`.

### C/C++

* hipFile C++ is **C++20** (`AIS_CXX_STANDARD`, default 20). Public C
  headers are **C11**. Other (non-hipFile) C++ in the monorepo defaults
  to C++17.
* Use modern C++ idioms: smart pointers (`std::unique_ptr`,
  `std::shared_ptr`), `std::optional`, `std::variant`, RAII wrappers
  (e.g. `FileDescriptor`), and `enum class` over plain enums.
* Disable copy and move explicitly with `= delete` for non-copyable
  resource owners; mark overrides with `override` and trivial destructors
  with `= default`.
* Prefer brace-initialization for members in constructor initializer lists.
* Use the passkey idiom (`passkey.h`) to restrict who can construct
  registered objects.
* Do not use GNU extensions; code should be platform-independent.

### Python

* Minimum supported version is **Python 3.10**.
* Bindings are written in Cython (`_chipfile.pxd`, `_hipfile.pyx`) and
  built with `scikit-build-core`.
* Use `from __future__ import annotations` and type hints throughout;
  gate import-only typing behind `if TYPE_CHECKING:`.
* Suppress lint warnings narrowly and inline with a justification comment
  (e.g. `# pylint: disable=W0718  # Suppress exceptions in a dtor`), never
  project-wide.

### Shell scripts

* All shell scripts must pass [ShellCheck](https://www.shellcheck.net/)
  with no issues.
* Use `#!/usr/bin/env bash` as a platform-independent shebang.
* bash-isms are allowed, within reason.
* Include the copyright/SPDX header.

## Error Handling

* Internally, errors are represented as exceptions derived from the
  standard library (typically `std::runtime_error`), one small struct per
  error condition with a fixed message (see `buffer.h`).
* Catch narrowly (`catch (const std::system_error &e)`) and rethrow with
  `throw;` when you cannot handle the specific case.
* At the public C boundary, translate exceptions into `hipFileOpError_t`
  return codes. A return of `-1` indicates an underlying C/POSIX error and
  that `errno` is likely set.
* In Python, C errors are surfaced as `HipFileException`.

## Testing Conventions

* Tests use **GoogleTest** and **GoogleMock**.
* All new functionality MUST include automatable tests (unit, system,
  and/or integration).
* Bugfixes MUST include a proof-of-concept test that fails before the fix
  and passes after.
* Tests are tagged with CTest `LABELS` (`unit`, `stress`, `system`,
  `internal`). Run unit tests with `ctest -V -L "unit"`. System and
  stress tests may not run on every machine.
* Test files mirror the source tree; mock headers are prefixed with `m`
  (`mhip.h`, `mbuffer.h`, `mstream.h`).
* Where a framework forces a warning (global constructors from
  GoogleTest macros), suppress it locally with the project's
  `HIPFILE_WARN_*` macros rather than globally.

## Tooling and Enforcement

* **clang-format 18** — C/C++ formatting (`util/format-source.sh`).
* **clang-tidy** — C/C++ static analysis; code must be clean.
* **Sanitizers** — code must be clean under the sanitizers configured in
  `cmake/AISSanitizers.cmake` (ASAN for pointer changes, TSAN for
  concurrency changes).
* **Compiler warnings** — C++ must compile warning-free.
* **black** + **pylint** — Python formatting and linting.
* **ShellCheck** — shell scripts.
* **cmakelint** — CMake (`.cmakelintrc`).
* **codespell** — spelling (`.codespellrc`, `.codespellignore`).
