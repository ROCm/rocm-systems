# Test Diagnostic Improvements

This document describes the enhanced error handling and diagnostic capabilities added to rocprofiler-sdk tests.

## Overview

Test failures now provide:
- **Categorized errors** for quick identification of root cause
- **Contextual information** to understand what went wrong
- **Diagnostic suggestions** for self-diagnosis
- **CI-aware logging** to reduce noise while maintaining debuggability
- **Infrastructure validation** to catch setup issues before test execution

## Error Categories

All test failures are tagged with one of these categories:

- `[SETUP]` - Missing dependencies, environment configuration issues
- `[PERMISSIONS]` - File/device permission problems
- `[INFRASTRUCTURE]` - GPU unavailable, driver issues, hardware problems  
- `[DATA]` - Data validation failures, unexpected values
- `[TIMEOUT]` - Execution timeouts
- `[CORRUPTION]` - Corrupted output files or data structures

## For Python Tests

### Using Enhanced Assertions

```python
from pytest_utils import ErrorCategory, assert_with_diagnostic

def test_example(input_data):
    assert_with_diagnostic(
        len(data) > 0,
        ErrorCategory.DATA,
        "Expected non-empty kernel dispatch data",
        context={
            "actual_length": len(data),
            "expected_min": 1,
            "source_file": filename
        },
        suggestions=[
            "Check if GPU kernels executed during test",
            "Verify profiling was enabled for kernel dispatch",
            "Look for error messages in earlier test output"
        ]
    )
```

### Prerequisite Checks

Tests automatically validate infrastructure before execution:

```python
# In conftest.py
from pytest_utils import check_test_prerequisites

def pytest_configure(config):
    """Run prerequisite checks before any tests"""
    error = check_test_prerequisites()
    if error:
        pytest.exit(error, returncode=1)
```

This catches issues like:
- Missing ROCm installation
- No GPU devices available
- Driver not loaded
- Permission problems

### Logging Levels

Control verbosity with environment variable:

```bash
# Minimal (CI default)
export ROCPROFILER_TEST_LOG_LEVEL=MINIMAL

# Normal (local default)
export ROCPROFILER_TEST_LOG_LEVEL=NORMAL

# Verbose (debug)
export ROCPROFILER_TEST_LOG_LEVEL=VERBOSE
```

Use logging functions in tests:

```python
from pytest_utils import log_verbose, log_info

def test_complex_validation(data):
    log_verbose("Starting timestamp validation")  # Only in verbose mode
    log_info("Validating 1000 records")  # Suppressed in CI
    # ... test logic ...
```

## For C++ Tests

### Enhanced Error Macros

```cpp
#include "common/defines.hpp"
#include "common/hip_diagnostics.hpp"

// With diagnostics
HIP_API_CALL_DIAG(
    hipMalloc(&ptr, size),
    ERROR_TAG_INFRASTRUCTURE,
    "Ensure GPU has sufficient memory available. Check with rocm-smi."
);

// Or for rocprofiler APIs
ROCPROFILER_CALL_DIAG(
    rocprofiler_create_context(...),
    ERROR_TAG_SETUP,
    "Verify rocprofiler-sdk library is properly linked"
);
```

### Infrastructure Validation

```cpp
int main() {
    // Validate GPU runtime before test execution
    validate_hip_runtime("my-test-name");
    
    // ... rest of test ...
}
```

This provides detailed diagnostics if GPU is unavailable:

```
[INFRASTRUCTURE] Test 'my-test-name' requires GPU but none available
Diagnostic steps:
  1. Verify ROCm is installed: ls /opt/rocm
  2. Check GPU devices: rocminfo
  3. Ensure user has GPU access: groups | grep render
  4. Check driver loaded: lsmod | grep amdgpu
```

## Examples of Improved Error Messages

### Before

```
AssertionError: assert 0 > 0
```

### After

```
[DATA] Field 'kernel_dispatch' has insufficient data
Context:
  expected_min: 1
  actual: 0
  field: kernel_dispatch
Diagnostic steps:
  1. Verify test workload executed properly
  2. Check if profiler captured events
  3. Minimum expected: 1, got: 0
```

### Before

```
HIP error : hipErrorNoDevice
```

### After

```
[INFRASTRUCTURE] Failed to query HIP devices: hipErrorNoDevice
Diagnostic steps:
  1. Check if ROCm is installed: ls /opt/rocm
  2. Verify GPU devices: rocminfo
  3. Check amdgpu driver: lsmod | grep amdgpu
  4. Verify user permissions: groups | grep render
```

## CI Integration

Tests automatically detect CI environment and adjust:
- Reduced log verbosity
- Error messages focus on actionable diagnostics
- Prerequisite failures provide clear setup instructions

CI detection checks for:
- `CI` environment variable
- `CONTINUOUS_INTEGRATION`
- `JENKINS_HOME`
- `GITLAB_CI`

## Migration Guide

### Updating Existing Tests

1. **Python tests**: Replace simple `assert` with `assert_with_diagnostic`
2. **Add conftest checks**: Run `./update_conftests.sh` or manually add pytest_configure
3. **C++ tests**: Replace HIP_API_CALL with HIP_API_CALL_DIAG where diagnostics help
4. **Add validation**: Call `validate_hip_runtime()` in main() for HIP tests

### Gradual Migration

You can migrate incrementally:
- New tests should use diagnostic utilities from the start
- Existing tests can be updated when failures occur
- Focus on tests with frequent infrastructure failures first

## Best Practices

1. **Choose appropriate category**: Use INFRASTRUCTURE for hardware/driver issues, DATA for validation failures
2. **Provide context**: Include actual vs expected values, file names, indices
3. **Actionable suggestions**: Each suggestion should be a concrete diagnostic step
4. **Avoid log spam**: Use log_verbose() for detailed tracing, log_info() for milestones
5. **Test the error paths**: Verify your diagnostic messages are actually helpful

## Performance Impact

- **Minimal overhead**: Error handling only executes on failure paths
- **No test performance degradation**: Validation checks run once at startup
- **CI runtime unchanged**: Reduced logging actually speeds up log processing

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'pytest_utils'`:

```bash
# Ensure pytest_utils is in PYTHONPATH
export PYTHONPATH="$PWD/projects/rocprofiler-sdk/tests/pytest-packages:$PYTHONPATH"
```

### Too Verbose Output

```bash
export ROCPROFILER_TEST_LOG_LEVEL=MINIMAL
```

### Need More Details

```bash
export ROCPROFILER_TEST_LOG_LEVEL=VERBOSE
```

## Files Modified

### New Files
- `pytest-packages/pytest_utils/diagnostics.py` - Core diagnostic utilities
- `common/hip_diagnostics.hpp` - C++ HIP diagnostic helpers
- `update_conftests.sh` - Helper script for adding prerequisite checks

### Enhanced Files
- `common/defines.hpp` - Added error category tags and diagnostic macros
- `kernel-tracing/validate.py` - Enhanced with diagnostic assertions
- `kernel-tracing/conftest.py` - Added prerequisite validation
- `async-copy-tracing/validate.py` - Enhanced error messages
- `async-copy-tracing/conftest.py` - Added prerequisite validation
- `bin/vector-operations/vector-ops.cpp` - Added GPU validation
- `pytest-packages/pytest_utils/__init__.py` - Exported diagnostic utilities

## Future Enhancements

Potential improvements for future iterations:
- Automatic log file analysis for common error patterns
- Integration with test result databases for failure trending
- Automated diagnostic script generation from error patterns
- Performance regression detection with categorized failures
