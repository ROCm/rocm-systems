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

To run the rocprofiler-sdk test suite alongside the building rocprofiler-sdk:

```bash
cmake -B build-rocprofiler-sdk -DROCPROFILER_BUILD_TESTS=ON -DROCPROFILER_BUILD_SAMPLES=ON .
cmake --build build-rocprofiler-sdk --target all --parallel 12
cd build-rocprofiler-sdk
ctest --output-on-failure -O ctest.all.log
```

In the above `ctest` command, `--output-on-failure` shows the test log only when the test fails and `-O <filename>` writes the log to a file in addition echoing it to the terminal.
CTest supports various options such as `-R` and `-E` for filtering which tests are run based on the test names, options such as `-L` and `-LE` for filtering which tests are run based on the test labels (Use `--print-labels` to see list of test labels).
Other useful options are `--rerun-failed`, `--stop-on-failure`, `--repeat until-fail:<N>`, `--show-only` (`-N`), `--verbose` (`-V`), and `--extra-verbose` (`-VV`).
Running `ctest -N -V` will show all details of the tests (command, environment, etc.) without running them.

One can also use [source/scripts/run-ci.py](./source/scripts/run-ci.py) locally with the argument `--disable-cdash` to avoid submitting the job to the CDash dashboard.
Examples using [run-ci.py](./source/scripts/run-ci.py) can be found in the [GitHub Actions workflows for rocprofiler-sdk](../../.github/workflows/rocprofiler-sdk-continuous_integration.yml).

If attempting to reproduce the sanitizer jobs, e.g. `cmake -DROCPROFILER_MEMCHECK=ThreadSanitizer ...`, locally instead of using [source/script/run-ci.py](./source/scripts/run-ci.py),
use [source/scripts/setup-sanitizer-env.sh](./source/scripts/setup-sanitizer-env.sh) to set the same sanitizer environment variables that rocprofiler-sdk uses during CI.

If trying to debug a specific test, use `ctest -N -V -R <test-name>` and use the output to create a bash script to run it, e.g. `ctest -N -V -R rocprofv3-test-trace-execute` produces:

```console
# ... removed for brevity

204: Test command: /home/user/rocm-systems/projects/rocprofiler-sdk/build-rocprofiler-sdk/bin/hip-graph
204: Working Directory: /home/user/rocm-systems/projects/rocprofiler-sdk/build-rocprofiler-sdk/tests/hip-graph-tracing
204: Environment variables:
204:  LD_PRELOAD=/home/user/rocm-systems/projects/rocprofiler-sdk/build-rocprofiler-sdk/lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so.0.0.0
204:  ROCPROFILER_TOOL_OUTPUT_FILE=hip-graph-tracing-test.json
204:  LD_LIBRARY_PATH=/home/user/rocm-systems/projects/rocprofiler-sdk/build-rocprofiler-sdk/lib:/usr/lib64:/usr/lib:/usr/local/lib
204:  ROCPROFILER_TOOL_CONTEXTS=HIP_API_CALLBACK,HIP_API_BUFFERED,KERNEL_DISPATCH_CALLBACK,KERNEL_DISPATCH_BUFFERED,CODE_OBJECT
Labels: integration-tests
  Test #204: test-hip-graph-tracing-execute

Total Tests: 1
```

Using all of the lines prefixed with `204:`, a bash script can be easily created:

```bash
# taken from "Environment variables:"
export LD_PRELOAD=/home/user/rocm-systems/projects/rocprofiler-sdk/build-rocprofiler-sdk/lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so.0.0.0
export ROCPROFILER_TOOL_OUTPUT_FILE=hip-graph-tracing-test.json
export LD_LIBRARY_PATH=/home/user/rocm-systems/projects/rocprofiler-sdk/build-rocprofiler-sdk/lib:/usr/lib64:/usr/lib:/usr/local/lib
export ROCPROFILER_TOOL_CONTEXTS=HIP_API_CALLBACK,HIP_API_BUFFERED,KERNEL_DISPATCH_CALLBACK,KERNEL_DISPATCH_BUFFERED,CODE_OBJECT

# taken from "Working Directory:"
pushd /home/user/rocm-systems/projects/rocprofiler-sdk/build-rocprofiler-sdk/tests/hip-graph-tracing

# taken from "Test command:" (and prefixed with `gdb --args` for debugging)
gdb --args /home/user/rocm-systems/projects/rocprofiler-sdk/build-rocprofiler-sdk/bin/hip-graph
```

If the test command uses [rocprofv3](./source/bin/rocprofv3.py), using debuggers such as `gdb` will require replacing prefixing with `gdb --args python3 /path/to/rocprofv3 ...`.
If rocprofv3 requires application replay, execute `set follow-fork-mode child` within the GDB command line prompt.

### Test Locations ###

* Integration tests are located in the top-level [tests](./tests) directory.
* Unit tests are located in a `tests` subdirectory of the units being tested.
* Samples are located in the top-level [samples](./samples) directory.
* Applications used for integration tests are located in the [tests/bin](./tests/bin) directory.

### Test Coding Style Guidelines ###

* Integration Test Applications ([tests/bin](./tests/bin))
  * These applications are a common suite of applications which can be used by any integration test.
  * These applications should, when possible, support command-line arguments to control the number of threads, streams, problem size, etc.
  * It is highly recommended to make use of threads, streams, etc. in the applications... few real-world applications are single-threaded and use only the default HIP stream.
* Integration tests
  * Should be composed of at least two tests: (1) an "execute" test which runs the profiler on the application and (2) a "validate" test
  * Pay attention to the naming conventions of the folders, files, and test names
  * The "validate" written in Python with PyTest, which validates the data collected during the "execute" phase. The Python script should be named `validate.py` and should be accompanied by a `conftest.py` and `pytest.ini`.
  * The `validate.py` main should return the following: `return pytest.main(["-x", __file__] + sys.argv[1:])`
  * In general, follow the same recipe as other integration tests
* Samples should be kept as simple as possible when possible: a `main.cpp` with a sample test application and a `client.cpp` which contains the tool built to demonstrate the functionality of the sample.
  * Please use existing samples such as [samples/api_buffered_tracing](./samples/api_buffered_tracing/), [samples/api_callback_tracing](./samples/api_callback_tracing/), [samples/external_correlation_id_request](./samples/external_correlation_id_request/), and [samples/intercept_table](./samples/intercept_table/) as a guide.
* Unit tests should follow the standard recipe:
  * Written with `GTest`
  * The first parameter to `TEST(<group>, <name>)` or `TEST_F(<group>, <name>)` should either be the name of the file, e.g. `TEST(agent, <name>)` in `agent.cpp`, or the name of test executable.
  * If the unit test is limit to a certain source file, e.g. `source/lib/common/utility.cpp`, then unit tests in the tests folder should be in a file by the same name, e.g. `source/lib/common/tests/utility.cpp`.
  * All of the source files in a unit test folder should be compiled into one executable and CTests should be added via `gtest_add_tests(...)`
  * It is permitted to deactive clang-tidy for unit tests via `rocprofiler_deactivate_clang_tidy()`
  * The `add_subdirectory(tests)` in parent directory's `CMakeLists.txt` should be guarded with `if(ROCPROFILER_BUILD_TESTS)`

## Code License ##

All code contributed to this project will be licensed under the license identified in the [LICENSE.md](LICENSE.md). Your contribution will be accepted under the same license.

## Release Cadence ##

Any code contribution to this library will be released with the next version of ROCm if the contribution window for the upcoming release is still open. If the contribution window is closed but the PR contains a critical security/bug fix, an exception may be made to include the change in the next release.
