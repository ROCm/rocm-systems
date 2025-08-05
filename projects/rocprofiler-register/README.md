# rocprofiler-register

## Overview

The rocprofiler-register library is a helper library that coordinates the modification of the intercept API table(s) of the HSA/HIP/ROCTx
runtime libraries by the ROCprofiler (v2) library. The purpose of this library is to provide a consistent and automated mechanism
of enabling performance analysis in the ROCm runtimes which does not rely on environment variables or unique methods for each runtime
library.

When a runtime is initialized (either explicitly and lazily) and the intercept API table is constructed, it passes this API table to
rocprofiler-register. Rocprofiler-register scans the symbols in the address space and if it detects there is at least one visible symbol named
`rocprofiler_configure` (which is a function provided by tools), it passes the intercept API table to the rocprofiler library (dlopening
the rocprofiler library if it is not already loaded). The rocprofiler library then does an extensive scan for _all_ the instances of
the `rocprofiler_configure` symbols and invokes each of them. The `rocprofiler_configure` function (again, provided by a tool) returns
effectively tells rocprofiler which behaviors it wants to be notified about, features it wants to use (e.g. API tracing, kernel dispatch timing),
etc.

## Environment Variables

| Environment Variable              | Description                                                               | Default Value  |
|-----------------------------------|---------------------------------------------------------------------------|----------------|
| `ROCP_TOOL_LIBRARIES`             | List of rocprofiler-sdk tool libraries (space, comma, or colon separated) | Empty (string) |
| `ROCPROFILER_REGISTER_ENABLED`    | Set to 0/false/no to disable rocprofiler-register                         | true (bool)    |
| `ROCPROFILER_REGISTER_SECURE`     | Additional checks to ensure authenticity of runtime libraries             | false (bool)   |
| `ROCPROFILER_REGISTER_FORCE_LOAD` | Load rocprofiler-sdk library regardless of whether there is a tool or not | false (bool)   |

## Contributing

All pull-requests should target the `amd-staging` branch.

### Creating a feature branch

```console
# fetch any updates
git fetch origin

# switch to staging branch
git checkout amd-staging

# update your copy of the staging branch
git pull --rebase

# create your feature branch off of amd-staging
git checkout -b <feature-branch>
```

In the event, your local clone of the repo has a `amd-staging` branch that diverges from the upstream branch,
do a hard reset of your local branch to match the upstream branch: `git reset --hard origin/amd-staging`.
Theoretically, you should never need to do this for `amd-mainline` but this can be applied to that
branch as well.

### Pulling in updates to `amd-staging` to your feature branch

Linear histories are preferred so if another PR is merged into `amd-staging` while your PR is still open, please
select the "Update with rebase" option (i.e. try to avoid a merge commit). From the command line, the git command
would be `git pull --rebase origin amd-staging`.

## Build and Installation

rocprofiler-register has a standard CMake build and install process. E.g. the following configure
rocprofiler-register to build with optimizations and without debug info in a `build-rocp-reg` subdirectory,
build using 4 jobs, and install to `/opt/rocprofiler-register`:

```console
cmake -B build-rocp-reg . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/rocprofiler-register
cmake --build build-rocp-reg --target all --parallel 4
cmake --build build-rocp-reg --target install
```
