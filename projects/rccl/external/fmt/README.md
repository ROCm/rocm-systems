# {fmt} 10.2.1 (header-only)

Vendored copy of the `{fmt}` headers used by RCCL (`fmt::format`). Every RCCL
build uses this copy; a system `fmt` package is never picked up.

- Upstream: https://github.com/fmtlib/fmt/tree/10.2.1
- License: MIT (see `LICENSE`)
- CMake target: `fmt::fmt-header-only`

Only `include/fmt` is checked in. Do not run fmt's own tests or install rules
from this tree.

To move to another release, replace `include/fmt` and `LICENSE` with the files
from the corresponding upstream tag and update the version in this file, in
`CMakeLists.txt`, and in `NOTICES.txt`.
