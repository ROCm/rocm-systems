#!/bin/bash

CURRENT_DIR="$( dirname -- "$0"; )";


echo -e "Running Profiler Tests"

echo -e "Running Unit tests for rocprofiler"
eval ${CURRENT_DIR}/tests/unittests/core/runCoreUnitTests
eval ${CURRENT_DIR}/tests/unittests/profiler/runProfilerUnitTests

echo -e "Running Feature Tests for diff Applicaitons;i.e: HSA,HIP,OpenMP,MPI"
eval ${CURRENT_DIR}/tests/featuretests/profiler/runFeatureTests

echo -e "Running Functional Tests; i.e: Load/Unload, Stress Tests"
eval ${CURRENT_DIR}/tests/featuretests/profiler/runFunctionalTests

echo -e "Running Standalone Tests"
echo -e "Warning: Some of these tests are path dependent.Please comment out next line if it fails"
eval ${CURRENT_DIR}/tests/featuretests/profiler/run_discrete_tests.sh

echo -e "Running Tracer Tests"
eval ${CURRENT_DIR}/tests/featuretests/tracer/runTracerFeatureTests