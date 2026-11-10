# nlohmann/json 3.12.0 (single header)

Vendored copy of the nlohmann/json single-header release used by RCCL's host
test tooling to read and write Google Test JSON reports. Every RCCL build uses
this copy; a system `nlohmann_json` package is never picked up.

- Upstream: https://github.com/nlohmann/json/tree/v3.12.0
- File: `single_include/nlohmann/json.hpp`
  (sha256 `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63`)
- License: MIT (see `LICENSE.MIT`)
- CMake target: `nlohmann_json::nlohmann_json`

Only `include/nlohmann/json.hpp` is checked in. Do not run the library's own
tests or install rules from this tree.

To move to another release, replace `include/nlohmann/json.hpp` and
`LICENSE.MIT` with the files from the corresponding upstream tag and update the
version and checksum in this file, in `CMakeLists.txt`, and in `NOTICES.txt`.
