---
description: "Use when: running tests, writing tests, checking test coverage, finding test files."
---
# Test Directories

| Suite | Path | Runner |
|-------|------|--------|
| C++ GTest | `tests/amd_smi_test/` | `build/tests/amd_smi_test/amdsmitst` |
| Python unit | `tests/python_unittest/unit_tests.py` | `python3` |
| Python integration | `tests/python_unittest/integration_test.py` | `python3` |
| Python CLI | `tests/python_unittest/cli_unit_test.py` | `python3` |
| Python perf | `tests/python_unittest/perf_tests.py` | `python3` |
| ABI checks | `tests/abi_check/` | CI workflow |
| API summary | `tests/api_summary.py` | `python3` |

All tests require GPU hardware. Python tests need `AMDSMI_PATH=/opt/rocm/share/amd_smi`.
