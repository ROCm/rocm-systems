## How to fork from us

To keep our development fast and conflict free, we recommend you to [fork](https://github.com/ROCm/rocSHMEM/fork) our repository and start your work from our `develop` branch in your private repository.

Afterwards, git clone your repository to your local machine. But that is not it! To keep track of the original develop repository, add it as another remote.

```
git remote add mainline https://github.com/ROCm/rocSHMEM.git
git checkout dev
```

As always in git, start a new branch with

```
git checkout -b topic-<yourFeatureName>
```

and apply your changes there.

## How to contribute to rocSHMEM

### Did you find a bug?

- Ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/ROCm/rocSHMEM/issues).

- If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/ROCm/rocSHMEM/issues/new).

### Did you write a patch that fixes a bug?

- Open a new GitHub [pull request](https://github.com/ROCm/rocSHMEM/compare) with the patch.

- Ensure the PR description clearly describes the problem and solution. If there is an existing GitHub issue open describing this bug, please include it in the description so we can close it.

- Ensure the PR is based on the `dev` branch of the rocSHMEM GitHub repository.

- rocSHMEM requires new commits to include a "Signed-off-by" token in the commit message (typically enabled via the `git commit -s` option), indicating your agreement to the project's [Developer's Certificate of Origin](https://developercertificate.org/) and compatibility with the project [LICENSE](https://github.com/ROCm/rocSHMEM/blob/main/LICENSE):


> (a) The contribution was created in whole or in part by me and I
> have the right to submit it under the open source license
> indicated in the file; or
> 
> (b) The contribution is based upon previous work that, to the best
> of my knowledge, is covered under an appropriate open source
> license and I have the right under that license to submit that
> work with modifications, whether created in whole or in part
> by me, under the same open source license (unless I am
> permitted to submit under a different license), as indicated
> in the file; or
> 
> (c) The contribution was provided directly to me by some other
> person who certified (a), (b) or (c) and I have not modified
> it.
> 
> (d) I understand and agree that this project and the contribution
> are public and that a record of the contribution (including all
> personal information I submit with it, including my sign-off) is
> maintained indefinitely and may be redistributed consistent with
> this project or the open source license(s) involved.

### Logging

rocSHMEM provides leveled logging macros defined in `src/log.hpp`. Use the appropriate level for your messages:

**Host macros** (for `__host__` code):
| Macro | Use for |
|---|---|
| `LOG_ERROR(fmt, ...)` | Non-fatal errors (always compiled) |
| `LOG_ERROR_EXIT(fmt, ...)` | Fatal errors, calls `exit()` (always compiled) |
| `LOG_ERROR_ABORT(fmt, ...)` | Fatal errors, calls `abort()` (always compiled) |
| `LOG_WARN(fmt, ...)` | Warnings (always compiled, runtime-gated) |
| `LOG_INFO(fmt, ...)` | Informational messages (always compiled, runtime-gated) |
| `LOG_API(fmt, ...)` | API call tracing (compile-gated by `BUILD_DEBUG_TRACE_HOST`) |
| `LOG_TRACE(fmt, ...)` | Internal traces (compile-gated by `BUILD_DEBUG_TRACE_HOST`) |

**Device macros** (for `__device__`/`__global__` code):
| Macro | Use for |
|---|---|
| `LOGD_ERROR(fmt, ...)` | Non-fatal errors (always compiled) |
| `LOGD_ERROR_ABORT(fmt, ...)` | Fatal errors, calls `abort()` (always compiled) |
| `LOGD_WARN(fmt, ...)` | Warnings (compile-gated by `BUILD_DEBUG_DEVICE`) |
| `LOGD_INFO(fmt, ...)` | Informational messages (compile-gated by `BUILD_DEBUG_DEVICE`) |
| `LOGD_API(fmt, ...)` | API call tracing (compile-gated by `BUILD_DEBUG_TRACE_DEVICE`) |
| `LOGD_TRACE(fmt, ...)` | Internal traces (compile-gated by `BUILD_DEBUG_TRACE_DEVICE`) |

**Guidelines:**
- Do not include trailing `\n` in format strings — the macros append it.
- Use `host::funcname (param=%p, pe=%d)` format for host API tracing.
- Use `device::funcname (ctx=%zd, dest=%p, pe=%d)` format for device API tracing.
- The `LOG_` macros produce a compile error if used in device code (and vice versa for `LOGD_`).
- Runtime verbosity is controlled by `ROCSHMEM_DEBUG_LEVEL` environment variable (default: `WARN`).

