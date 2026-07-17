---
description: Build with the coverage preset, run unit tests, generate a gcovr report
argument-hint: [label]
---

Label: `${1:-all}` — should describe the current work (feature/bugfix name),
not left as the default.

1. Configure: `cmake --preset coverage` (requires `gcovr>=8.4`:
   `pip install 'gcovr>=8.4'`).
2. Build the whole project: `cmake --build build/coverage -j$(nproc)` —
   never `--target`.
3. Run the aggregated unit-test binary directly (not `ctest`):
   `./build/coverage/bin/rocprof-sys-unit-tests`
4. Generate the report:
   `python3 scripts/generate-coverage.py --build-dir build/coverage --source-dir . --output-dir .codecov --label ${1:-all}`
5. Report the summary from `.codecov/${1:-all}-summary.json` (coverage_pct,
   covered_lines, total_lines, function_pct, file_count).

See the `rocprofsys-coverage` section of the `rocprofsys` skill and
`testing-conventions` rule for details and common mistakes.
