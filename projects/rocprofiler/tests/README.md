# ROCProfiler testing environment.

This document explains how ROCProfiler testing environment works.
We make use of the GoogleTest (Gtest) framework to automatically find and add test cases to the CMAKE testing environment.

# Test Categories

ROCProfiler testing is categorised as following:

- unittests (Gtest Based)
- featuretests (Gtest Based)
- memorytests (standalone)
- performancetests (TBD)

### Quickstart

ROCProfiler tests are integrated into the top-level cmake project. The tests depend upon the installed version of ROCProfiler.
Typical usage (paths relative to top of the ROCProfiler repo):
```
$ ./build.sh
$ ./rocprofv2 -ct
```

### How to add a new test

The test infrastructure use a hierarchy of folders. So add the new test to the appropriate folder. 
The tests/unittests/session/session_gtest.cpp file contains a simple unit test and is a good starting point for other tests.
Copy this to a new test name and modify it.

### Run subsets of all tests:
```

# Run unit tests on the commandline
./build/tests/unittests/runUnitTests

# Run profilerfeaturetests on the commandline
./build/tests/featuretests/profiler/runFeatureTests

# Run tracer featuretests on the commandline
./build/tests/featuretests/tracer/runTracerFeatureTests

# Run all tests:
./rocprofv2 -t OR ./rocprofv2 -ct
               OR
 make -j check OR ./run_tests.sh

```

### Performance tests:
```
TBD

# Guidelines for adding new tests

- Prefer to enhance an existing test as opposed to writing a new one. Tests have overhead to start and many small tests spend precious test time on startup and initialization issues.
- Make the test run standalone without requirement for command-line arguments.  This makes it easier to debug since the name of the test is shown in the test report and if you know the name of the test you can the run the test.

