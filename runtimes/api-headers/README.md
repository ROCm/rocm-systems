# Runtime API Headers

This directory contains static, dependency-free API headers intended for both
C++ consumers and Rust API bindings.

The headers are organized by API family under `include/`:

- `amdf`: AMD Framework API headers.
- `hsa`: Heterogeneous System Architecture API headers.
- `uapi`: Linux userspace API headers used by ROCm runtimes.

These headers are not yet exposed through a CMake target. As runtime components
begin consuming them from this repository, build integration will be added to
export the target and install the headers.
