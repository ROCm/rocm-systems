# CMake standards

- **Build tree** — Out-of-source builds only; in-source configure is forbidden by root `CMakeLists.txt`.
- **Targets** — Prefer target-based `target_*` APIs; avoid broad `include_directories` unless matching existing project patterns.
- **ROCm** — Respect `ROCM_PATH` / `CMAKE_PREFIX_PATH` conventions already set in project CMake files.
- **Changes** — Keep edits localized; do not rename install targets or package metadata without maintainer coordination.
