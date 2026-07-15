---
name: amdsmi-test-runner
description: "Run C++ and Python tests for amd-smi. Use when: running tests, verifying test results, checking test coverage, pre-review test validation."
---

# Test Runner — amd-smi

How to build and run all test suites for amd-smi. Assumes the `amdsmi-build-install` skill has already been executed (build exists and is installed).

## Prerequisites

- amd-smi built and installed (run `amdsmi-build-install` skill first)
- AMD GPU hardware available for integration tests
- `build/` directory exists with compiled test binaries

## C++ Tests (GTest)

### Build Test Binary

If not already built (the build skill builds with `-DBUILD_TESTS=ON`):

```bash
cd build && make -j$(nproc) amdsmitst
```

### Run Tests

```bash
cd build/tests
source ../../tests/amd_smi_test/amdsmitst.exclude
source ../../tests/amd_smi_test/detect_asic_filter.sh
./amdsmitst --gtest_filter="-${GTEST_EXCLUDE}"
```

### Interpret Results

| Output | Meaning | Severity |
|--------|---------|----------|
| `[  PASSED  ]` | Test passed | — |
| `[  FAILED  ]` | Test failed | ❌ BLOCKING |
| `[  SKIPPED ]` | Test skipped (blacklisted) | — |
| Binary not found | Build didn't produce test binary | ⚠️ IMPORTANT |

## Python Tests

### Unit Tests

```bash
cd tests/python_unittest
python3 -m pytest -v
```

<!-- ### CLI Tests

```bash
cd amdsmi_cli
python3 -m pytest -v
``` -->

### Import Verification

```bash
# System install
python3 -c "import amdsmi; print(amdsmi.__version__)"

# Verify CLI works
amd-smi version
amd-smi static
```

## Test Contexts

Tests must work in **both** install contexts:

| Context | How to Verify |
|---------|--------------|
| System install (RPM/DEB) | Default after `amdsmi-build-install` |

## Running All Tests (One-Shot)

```bash
cd build/tests && \
source ../../tests/amd_smi_test/amdsmitst.exclude && \
source ../../tests/amd_smi_test/detect_asic_filter.sh && \
./amdsmitst --gtest_filter="-${GTEST_EXCLUDE}" && \
cd ../../tests/python_unittest && python3 -m pytest -v
```

## Output

On success, report:
- **C++ tests:** X passed, Y failed, Z skipped
- **Python tests:** X passed, Y failed
- **Any new test failures** compared to baseline

On failure:
- **Which suite failed** (C++ GTest, Python unit, CLI)
- **Specific test names** that failed
- **Test output** for failed tests
- Test failures are **❌ BLOCKING** for the review
