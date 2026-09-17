# rocprofiler-register

## Overview

The rocprofiler-register library is a helper library that coordinates the modification of the intercept API table(s) of the HSA/HIP/ROCTx
runtime libraries by the ROCprofiler (v2) library. The purpose of this library is to provide a consistent and automated mechanism
of enabling performance analysis in the ROCm runtimes which does not rely on environment variables or unique methods for each runtime
library.

When a runtime is initialized (either explicitly or lazily) and constructs its intercept API table,
it passes the table to rocprofiler-register. Rocprofiler-register scans the process for
`rocprofiler_configure` symbols and, when one is present or startup loading was explicitly
requested, passes the table to rocprofiler-sdk. Rocprofiler-sdk discovers and invokes the
tool-provided configuration functions, which specify requested services such as API tracing or
kernel dispatch timing.

Attachment capability is independent of startup profiling. When attachment is enabled through
`ROCP_TOOL_ATTACH=1` or the corresponding build default, rocprofiler-register initializes the
attachment helper before propagating the first supported runtime, even if rocprofiler-sdk already
has startup or anytime clients. HSA-dependent queue setup is completed when HSA registers. A later
runtime attachment explicitly loads its tool as an additional rocprofiler-sdk client through
anytime initialization. Detaching that client does not stop existing clients.

## Environment Variables

| Environment Variable              | Description                                                               | Default Value                            |
|-----------------------------------|---------------------------------------------------------------------------|------------------------------------------|
| `ROCP_TOOL_LIBRARIES`             | List of rocprofiler-sdk tool libraries (space, comma, or colon separated) | Empty (string)                           |
| `ROCP_TOOL_ATTACH`                | Initialize runtime-attachment support during runtime registration         | Build-dependent                          |
| `ROCPROFILER_REGISTER_LIBRARY`    | Explicit rocprofiler-sdk library to load                                  | Empty (string)                           |
| `ROCPROFILER_REGISTER_ENABLED`    | Set to 0/false/no to disable rocprofiler-register                         | true (bool)                              |
| `ROCPROFILER_REGISTER_SECURE`     | Additional checks to ensure authenticity of runtime libraries             | false (bool)                             |
| `ROCPROFILER_REGISTER_FORCE_LOAD` | Load rocprofiler-sdk at startup regardless of visible tools               | false (bool)                             |

## Contributing

The default branch is `develop`.

> _**All pull-requests should target the `develop` branch**_

### Creating a feature branch

```console
# fetch any updates
git fetch origin

# switch to development branch
git checkout develop

# update your copy of the development branch
git pull --rebase

# create your feature branch off of develop branch
git checkout -b <feature-branch>
```

In the event, your local clone of the repo has a `develop` branch that diverges from the upstream branch,
do a hard reset of your local branch to match the upstream branch: `git reset --hard origin/develop`.

### Pulling in updates to `develop` to your feature branch

Linear histories are preferred so if another PR is merged into `develop` while your PR is still open, please
select the "Update with rebase" option (i.e. try to avoid a merge commit). From the command line, the git command
would be `git pull --rebase origin develop`.

## Build and Installation

rocprofiler-register has a standard CMake build and install process. E.g. the following configure
rocprofiler-register to build with optimizations and without debug info in a `build-rocp-reg` subdirectory,
build using 4 jobs, and install to `/opt/rocprofiler-register`:

```console
cmake -B build-rocp-reg . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/rocprofiler-register
cmake --build build-rocp-reg --target all --parallel 4
cmake --build build-rocp-reg --target install
```
