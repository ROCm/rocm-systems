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
Your pull request should target the default branch. Our current default branch is the **develop**
branch, which serves as our integration branch.

All changes must meet the following requirements for review/acceptance:

1. All C and C++ code must be formatted with clang-format-18, using the `.clang-format` at the
   profiler-hub root.
2. All CMake code must be formatted with gersemi (version 0.25.1).
3. All text files must end with the new line character.
4. C++ changes should pass the clang-tidy checks configured in the `.clang-tidy` at the profiler-hub
   root.
5. Compiler warnings introduced on library-target code should be addressed.

clang-format and gersemi are enforced during CI; a formatting difference fails the check. clang-tidy
also runs in CI and reports its findings on the pull request, but a clang-tidy finding does not by
itself fail the job — treat new findings on touched code as something to fix before merge.

See [STYLEGUIDE.md](STYLEGUIDE.md) for the coding conventions and for how to run the formatters and
static analysis locally.

## Coding Style ##

profiler-hub's coding conventions — file and directory layout, naming, comments and documentation,
error handling, CMake usage, and formatting/tooling — live in [STYLEGUIDE.md](STYLEGUIDE.md).

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
* [STYLEGUIDE.md](STYLEGUIDE.md) — profiler-hub coding style and formatting/tooling conventions.
