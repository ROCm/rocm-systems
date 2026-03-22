# C++ standards (rocprofiler-compute native tool)

- **Standard** — C++17 required (`src/lib/CMakeLists.txt`).
- **Dependencies** — Link against `rocprofiler-sdk` via CMake `find_package` as in existing `CMakeLists.txt`; do not introduce new libraries without approval.
- **Scope** — Keep `src/lib/` changes minimal and consistent with `rocprofiler_compute_tool.cpp` / `helper.cpp` style.
- **Safety** — Prefer RAII and clear ownership; avoid unnecessary heap churn on hot paths.
- **Verify** — Configure out-of-source build and compile the targets you touch (see root `CMakeLists.txt`).
