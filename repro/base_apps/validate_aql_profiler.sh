#!/bin/bash
# AQL Profiler Validation Script

echo 'AQL Profiler Validation'
echo '======================'

# Check if executables exist and are executable
for exe in aql_profiler_test app_with_aql_profiling; do
    if [[ -x $exe ]]; then
        echo "✓ $exe is executable"
    else
        echo "✗ $exe not found or not executable"
        exit 1
    fi
done

# Run error handling test (should not require GPU)
echo 'Running error handling test...'
./aql_profiler_test error

# Check library exists
if [[ -f libaql_profiler.a ]]; then
    echo "✓ libaql_profiler.a exists"
    echo "Library size: $(ls -lh libaql_profiler.a | awk '{print $5}')"
else
    echo "✗ libaql_profiler.a not found"
    exit 1
fi

echo 'Validation complete - AQL profiler ready for use'
