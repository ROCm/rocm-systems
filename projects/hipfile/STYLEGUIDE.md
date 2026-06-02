# hipFile Style Guide

## File and Directory Structure
### Organization
### Naming

* Use hyphens instead of underscores in file and directory names

## Formatting

* Code must meet the requirements of our options to `clang-format`

## Naming Conventions

## Comments and Documentation

## Header Guidelines

* Headers should follow [include-what-you-use](https://include-what-you-use.org/) guidelines
* Use `#pragma once` instead of include guards
* Place local headers (quoted) ahead of system headers (brackets), with each block in alphabetical order
* Public headers MUST have Doxygen markup for all public symbols
* Private headers should have Doxygen markup flagged with `@internal`

## Language Features and Idioms

### CMake
* We use CMake 3.x (see the root `CMakeLists.txt` for the specific version) and have not moved to 4.x yet
* Use modern CMake (e.g., 3.x) paradigms and avoid "legacy CMake"

### C/C++

* C++17 and C11 are supported
* Use modern C++ idioms
* Do not use GNU extensions
* Code should be platform-independent

### Shell scripts

* All shell scripts should be run through [ShellCheck](https://www.shellcheck.net/) and report no issues
* Use `#!/usr/bin/env bash` as a platform-independent shebang line
* bash-isms are allowed, within reason

## Error Handling

## Testing Conventions

* All new functionality introduced MUST include tests of the feature
    * This could include unit tests, system tests, and integration tests
    * These tests MUST be automatable
* Bugfixes MUST include a proof-of-concept test that fails before the fix and passes after

## Tooling and Enforcement
clang-tidy, formatter, etc.
