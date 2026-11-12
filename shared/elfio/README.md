# ELFIO

The one maintained copy of [ELFIO](https://github.com/serge1/ELFIO), the
header-only ELF reader/writer, for rocm-systems projects.

ELFIO has had no tagged release since 3.12 (2023), and upstream fixes can take a
long time to merge. Fixes therefore land here as patches on top of a pinned
upstream commit, and every consumer picks them up from this one place.

## Contents

| Path | What it is |
| --- | --- |
| `SOURCE` | Upstream repository and pinned commit (`<url> <commit>`) |
| `patches/` | Fixes applied on top of `SOURCE`, in name order |
| `elfio/`, `LICENSE.txt` | Generated: upstream at `SOURCE` with `patches/` applied |
| `update.py` | Regenerates or verifies the generated files |
| `CMakeLists.txt` | Header-only `rocm::elfio` target |

`elfio/` is generated. Never edit it by hand: change a patch, then run
`update.py`.

## Current patches

None. `elfio/` is upstream at `SOURCE` unchanged.

## Using it

```cmake
add_subdirectory(<path-to>/shared/elfio ${CMAKE_CURRENT_BINARY_DIR}/rocm-elfio)
target_link_libraries(<target> PRIVATE rocm::elfio)
```

Guard the `add_subdirectory()` with `if(NOT TARGET rocm-elfio)` if more than
one part of a project may add it. Nothing is installed: ELFIO stays a private,
build-time dependency of each consumer.

A sparse checkout of a consuming project must include `shared/elfio`.

Consumers:

- `projects/clr` (ROCclr's `amd::Elf`, used by the HIP and OpenCL runtimes)
- `projects/rocprofiler-sdk` (unless `ROCPROFILER_BUILD_ELFIO=OFF`)

## Updating

```bash
# Verify the generated files match SOURCE + patches
python3 shared/elfio/update.py --check

# Regenerate after adding or changing a patch
python3 shared/elfio/update.py

# Move to a newer upstream commit; SOURCE records its full SHA
python3 shared/elfio/update.py --commit <sha-or-ref>
```

Each patch names its upstream pull request. Before moving `SOURCE` to an
upstream commit that contains a patch, delete that patch: `update.py` applies
every patch to the new commit, and an already-merged one fails to apply.

To add a fix, open it upstream first if possible, then add the change here as
the next numbered patch with an `Upstream:` line pointing at the pull request.
