# Contributing to ROCProfiler SDK #

Contributions are welcome. Contributions at a basic level must conform to the MIT license and pass code test requirements (i.e. ctest). The author must also be able to respond to comments/questions on the PR and make any changes requested.

## Issue Discussion ##

Please use the GitHub Issues tab to let us know of any issues.

* Use your best judgment when creating issues. If your issue is already listed, please upvote the issue and
comment or post to provide additional details, such as the way to reproduce this issue.
* If you're unsure if your issue is the same, err on caution and file your issue.
  You can add a comment to include the issue number (and link) for a similar issue. If we evaluate
  your issue as being the same as the existing issue, we'll close the duplicate.
* If your issue doesn't exist, use the issue template to file a new issue.
  * When you file an issue, please provide as much information as possible, including script output, so
we can get information about your configuration. This helps reduce the time required to
    reproduce your issue.
  * Check your issue regularly, as we may require additional information to reproduce the
issue successfully.
* You may also open an issue to ask the maintainers whether a proposed change
meets the acceptance criteria or to discuss an idea about the library.

## Acceptance Criteria ##

Github issues are recommended for any significant change to the code base that adds a feature or fixes a non-trivial issue.
If the code change is large without the presence of an issue (or prior discussion with AMD), the change may not be reviewed.
Small fixes that fix broken behavior or other bugs are always welcome with or without an associated issue.

## Pull Request Guidelines ##

By creating a pull request, you agree to the statements made in the [code license](#code-license) section.
Your pull request should target the default branch. Our current default branch is the **develop** branch, which serves as our integration branch.

All changes must meet the following requirements for review/acceptance:

1. All C and C++ code must be formatted with clang-format-11.
2. All Python code must be formatted with black.
3. All CMake code must be formatted with cmake-format.
4. All C++ changes must pass the clang-tidy checks (clang-tidy version 15.x.x through version 19.x.x are acceptable).
5. All text files must end with the new line character.
6. All C and C++ compiler warnings must be fixed

All the above checks are enforced during CI.
The [requirements.txt](requirements.txt) defines the exact versions of formatters and linters as needed.

In order to streamline requirements 1-4, support has been built into the rocprofiler-sdk build system.
By default, CMake will search for `clang-format`, `black`, and `cmake-format`. If `clang-format` is found,
CMake will add a `format-source` build target, e.g. `make format-source`; if `black` is found, CMake
will add a `format-python` build target; if `cmake-format` is found, CMake will add a `format-cmake` build
target. If any of the `format-source`, `format-python`, or `format-cmake` targets exist, CMake will
also add a generic `format` build target which depends on all the available `format-*` targets. Thus,
running `make format` will apply formatting to C, C++, Python, and CMake. The CMake option
`ROCPROFILER_ENABLE_CLANG_TIDY` can be used to enable clang-tidy checks when compiling the source code.

For requirement #5, it is recommended to configure your IDE to automatically add new lines at the end of files.

For requirement #6, the CMake option `ROCPROFILER_BUILD_DEVELOPER` can be used to enable the `-Werror` compiler flag,
which treats warnings as errors.

For simplicity, rocprofiler-sdk provides a CMake option `ROCPROFILER_BUILD_CI` to enable the following CMake options by default:
`ROCPROFILER_BUILD_TESTS`, `ROCPROFILER_BUILD_SAMPLES`, `ROCPROFILER_BUILD_DEVELOPER`. However, if CMake is initially configured
with `ROCPROFILER_BUILD_CI=OFF` (the default), re-running cmake with `ROCPROFILER_BUILD_CI=ON` does not change the values of
`ROCPROFILER_BUILD_TESTS` and `ROCPROFILER_BUILD_SAMPLES` (which are also, by default, OFF).

Thus, the build setup for developer contributions is the following:

```bash
python3 -m pip install --user ./requirements.txt
cmake -B build-rocprofiler-sdk . -DROCPROFILER_BUILD_CI=ON -DROCPROFILER_ENABLE_CLANG_TIDY=ON
```

## Coding Style Guidelines ##

1. Use the file extension `.h` for C-compatible header files and `.c` for C implementation files.
2. Use the file extension `.hpp` for C++ header files and `.cpp` for C++ implementation files.
3. All public APIs which require linking must be compatible with C. Public C++ APIs may only be distributed as header-only implementations.
4. The source code organization within [source](./source) should roughly align to the installation locations, e.g. an executable `foo` which will be
  installed in `bin` should be in either `source/bin/foo.py` (if script which doesn't require compilation) or in the folder `source/bin/foo/` (if requires compilation).
5. In a `CMakeLists.txt` file, do not add sources to a target from any other directory other than the current directory; instead use a combination of `add_subdirectory` and `target_sources`.
6. In CMake, always use target-based semantics such as `target_include_directories(...)`, `target_compile_definitions(...)`; CMake functions which are not target-based such as `include_directories(...)`, `add_definitions(...)` should be strictly avoided.
7. In CMake, use of `INTERFACE` libraries is encouraged for compiler options, compiler definitions, include directories, etc.
8. In internal implementations, designs requiring internal communication across translation units should prefer procedural or functional interfaces instead of object-oriented interfaces.
    * E.g. headers should declare simple structs without any protected or private data and standalone functions returning or operating on the aforementioned structs instead of exposing classes with public/protected/private member variables and member functions.
    * Within the implementation file, classes may be used as desired.
9. All public API structs which as used in C should have a `uint64_t size` member variable as the first member variable. Tool developers use this for ABI-compatability checks at runtime when accessing a struct instance via a pointer.
    * In internal implementations, all public API structs should be initialized via the `init_public_api_struct` function defined in [source/lib/common/utility.hpp](./source/lib/common/utility.hpp).
    * If a public API struct is intentionally padded, the padding should be of the form `uint8_t reserved_padding[<num-bytes>]` at the end of the struct. The name `reserved_padding` is important to how `init_public_api_struct` sets the `.size` value. Furthermore, static asserts should be added to ensure that `sizeof(T)` is never changed.
10. In internal implementations, one variable should be initialized per line: `int x, y;` is not permitted. The preferred form of variable initialization for non-primitive types is `auto <name> = <type>{}`... in other words, `auto` on the LHS and curly braces `{}` instead of parentheses `()`.
    * The use of `auto` is for readability: determining the variable name in `auto val = std::unordered_map<Foo, std::unordered_map<uint64_t, std::vector<Bar>>{};` is quite a bit easier than in `std::unordered_map<Foo, std::unordered_map<uint64_t, std::vector<Bar>> val{};`.
    * The use of curly braces has many benefits: prevention of implicit casting, is not potentially ambiguous with a function call (i.e. `Foo()` in `auto val = Foo()` may be a function call or construction of an object of class `Foo` whereas `Foo{}` can only be construction of an object of class `Foo`), etc.

## Testing Guidelines ##

For the component-level testing strategy, including what to test at the unit, integration, performance, and coverage layers, how to run tests locally, and how to add new tests, see [TESTING.md](./TESTING.md).

For a standard local test run:

```bash
cmake -B build-rocprofiler-sdk -DROCPROFILER_BUILD_TESTS=ON -DROCPROFILER_BUILD_SAMPLES=ON .
cmake --build build-rocprofiler-sdk --target all --parallel 12
cd build-rocprofiler-sdk
ctest --output-on-failure -O ctest.all.log
```

Use CTest filters such as `-R`, `-E`, `-L`, and `-LE` for targeted runs.

## Code License ##

All code contributed to this project will be licensed under the license identified in the [LICENSE.md](LICENSE.md). Your contribution will be accepted under the same license.

## Release Cadence ##

Any code contribution to this library will be released with the next version of ROCm if the contribution window for the upcoming release is still open. If the contribution window is closed but the PR contains a critical security/bug fix, an exception may be made to include the change in the next release.
