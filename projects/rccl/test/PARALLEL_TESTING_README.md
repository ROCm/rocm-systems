# GPU-Aware Parallel Test Execution for RCCL

## Quick Start

### Run tests with parallel GPU scheduling (recommended for 8 GPUs):

```bash
export UT_PARALLEL_TESTS=1
export UT_PARALLEL_VERBOSE=1  # Optional: see scheduling decisions
./rccl-UnitTests --gtest_filter="ParallelExecution.*"
```

### Compare with traditional sequential execution:

```bash
# Sequential (old way)
time UT_PARALLEL_TESTS=0 ./rccl-UnitTests --gtest_filter="ParallelExecution.BasicSweep"

# Parallel (new way)
time UT_PARALLEL_TESTS=1 ./rccl-UnitTests --gtest_filter="ParallelExecution.BasicSweep"
```

## Problem Statement

**Before**: Tests ran sequentially through GPU counts 1→2→3→4→5→6→7→8, leaving GPUs idle:
```
1 GPU test:  ████░░░░░░░░  (12.5% utilization, 7 GPUs idle)
2 GPU test:  ████████░░░░  (25% utilization, 6 GPUs idle)
4 GPU test:  ████████████  (50% utilization, 4 GPUs idle)
8 GPU test:  ████████████  (100% utilization, 0 GPUs idle)
Average:     ~30% GPU utilization
```

**After**: Multiple tests run concurrently when GPU resources permit:
```
Time 0:  [8-GPU test]────────────────────────  (all 8 GPUs busy)
Time 1:  [4-GPU test]──────[4-GPU test]──────  (all 8 GPUs busy)
Time 2:  [2-GPU][2-GPU][2-GPU][2-GPU]─────────  (all 8 GPUs busy)
Average: ~75-90% GPU utilization
Speedup: 3-5x depending on test distribution
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UT_PARALLEL_TESTS` | 0 | Enable (1) or disable (0) parallel execution |
| `UT_MAX_PARALLEL_TESTS` | 8 | Maximum concurrent test processes |
| `UT_PARALLEL_VERBOSE` | 0 | Enable detailed scheduling logs |
| `UT_MIN_GPUS` | 1 | Minimum GPU count for sweep |
| `UT_MAX_GPUS` | 8 | Maximum GPU count for sweep |

## Architecture

```
┌─────────────────────────────────────────┐
│      ParallelTestRunner API             │
│  (High-level test registration)         │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         GPUScheduler Engine             │
│  ┌────────────────────────────────────┐ │
│  │  Priority Queue (by GPU count)     │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  GPU Allocation Tracker            │ │
│  │  [GPU0][GPU1][GPU2]...[GPU7]       │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  Running Test Monitor (waitpid)    │ │
│  └────────────────────────────────────┘ │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      fork() + Environment Setup         │
│  - HIP_VISIBLE_DEVICES=0,1,2,3         │
│  - CUDA_VISIBLE_DEVICES=0,1,2,3        │
│  - ROCR_VISIBLE_DEVICES=0,1,2,3        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         Child Process                   │
│      (existing TestBed)                 │
└─────────────────────────────────────────┘
```

## Key Features

### 1. **Automatic GPU Assignment**
Each forked test process gets environment variables set to restrict GPU visibility:
```cpp
// Example: 4-GPU test assigned to GPUs 4-7
HIP_VISIBLE_DEVICES=4,5,6,7
CUDA_VISIBLE_DEVICES=4,5,6,7
ROCR_VISIBLE_DEVICES=4,5,6,7
```

### 2. **Priority-Based Scheduling**
Default priority = `100 - (numGPUs * 10)`:
- 8-GPU tests: priority 20 (scheduled first)
- 4-GPU tests: priority 60
- 1-GPU tests: priority 90 (scheduled last)

This "largest first" strategy minimizes GPU fragmentation.

### 3. **Process Isolation**
Each test runs in its own forked process:
- No GPU state sharing between tests
- Independent RCCL communicators
- Clean failure isolation

### 4. **Real-Time Monitoring**
- Tracks GPU busy/idle state
- Monitors test completion via `waitpid()`
- Calculates utilization statistics

## Usage Examples

### Example 1: Simple Integration with Existing Tests

```cpp
TEST(MyTest, WithParallelExecution)
{
    ParallelTestRunner runner;

    // Register tests for 1-8 GPUs
    for (int numGPUs = 1; numGPUs <= 8; ++numGPUs)
    {
        runner.registerTest("MyTest_" + std::to_string(numGPUs) + "GPU",
                          numGPUs,
                          [numGPUs]() {
                              TestBed testBed;
                              testBed.ev.minGpus = numGPUs;
                              testBed.ev.maxGpus = numGPUs;
                              // ... existing test code ...
                          });
    }

    ASSERT_TRUE(runner.executeAll());
}
```

### Example 2: Using TestSweepBuilder

```cpp
TEST(MyTest, AutomaticSweep)
{
    ParallelTestRunner runner;
    TestSweepBuilder builder(runner);

    builder.registerSweep("MyTest", 1, 8, [](int numGPUs) {
        return [numGPUs]() {
            // Test logic here
        };
    });

    ASSERT_TRUE(runner.executeAll());
}
```

### Example 3: Custom Priorities

```cpp
ParallelTestRunner runner;

// Critical test runs first
runner.registerTest("CriticalLargeTest", 8, testFunc, 1000);

// Regular tests
runner.registerTest("RegularTest", 4, testFunc);  // Auto priority
```

## Performance Expectations

### Test Distribution Analysis

Analyze your test suite:
```bash
# Count tests by GPU requirement
grep -r "numGpus" test/*.cpp | wc -l
```

### Expected Speedup

| Test Profile | Sequential Time | Parallel Time | Speedup |
|--------------|----------------|---------------|---------|
| Many 1-2 GPU tests | 100 min | 15-20 min | 5-7x |
| Balanced distribution | 80 min | 25-30 min | 3-4x |
| Mostly 8-GPU tests | 60 min | 50 min | 1.2x |

### GPU Utilization Targets

- Sequential execution: 20-35% average utilization
- Parallel execution: 70-90% average utilization
- Peak utilization during parallel: ~95%

## Debugging

### Enable Verbose Logging

```bash
export UT_PARALLEL_VERBOSE=1
./rccl-UnitTests --gtest_filter="ParallelExecution.BasicSweep"
```

Output shows:
```
GPUScheduler initialized with 8 GPUs, max 8 concurrent tests
Job submitted: AllReduce_8GPU (ID: 0, GPUs: 8, Priority: 20)
Job submitted: AllReduce_4GPU (ID: 1, GPUs: 4, Priority: 60)
...
Allocated GPUs: 0,1,2,3,4,5,6,7
Launching test: AllReduce_8GPU (Job ID: 0) on 8 GPUs
Child process 12345: HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
Test completed: AllReduce_8GPU (Duration: 5234 ms, Status: PASSED)
Released GPUs: 0,1,2,3,4,5,6,7
...
=============== GPU Scheduling Statistics ===============
Total tests executed: 8
Tests passed: 8
Tests failed: 0
Total execution time: 12456 ms
Average GPU utilization: 82.3%
========================================================
```

### Common Issues

#### Issue: Tests fail only in parallel mode

**Symptom**: Tests pass with `UT_PARALLEL_TESTS=0`, fail with `UT_PARALLEL_TESTS=1`

**Possible Causes**:
1. Shared state between tests (global variables)
2. Hardcoded GPU IDs instead of using environment variables
3. Race conditions in test setup/teardown

**Solution**:
```bash
# Run with verbose logging to see GPU assignments
UT_PARALLEL_VERBOSE=1 UT_PARALLEL_TESTS=1 ./rccl-UnitTests

# Run suspected test in isolation
UT_PARALLEL_TESTS=1 ./rccl-UnitTests --gtest_filter="SuspectedTest"

# Verify environment variables are respected
# Add debug output in test:
INFO("HIP_VISIBLE_DEVICES=%s\n", getenv("HIP_VISIBLE_DEVICES"));
```

#### Issue: Out of memory

**Symptom**: Tests crash with OOM errors

**Solution**:
```bash
# Reduce concurrent tests
export UT_MAX_PARALLEL_TESTS=4

# Or disable parallelism for memory-intensive tests
export UT_PARALLEL_TESTS=0
```

#### Issue: Inconsistent GPU assignments

**Symptom**: Test expects certain GPU topology

**Solution**: Use `registerTestWithGPUInfo` to know assigned GPUs:
```cpp
runner.registerTestWithGPUInfo("TopologyTest", 4,
    [](const std::vector<int>& gpus) {
        // gpus contains actual physical GPU IDs
        // Test can verify topology if needed
    });
```

## Migration Strategy

### Phase 1: Add Parallel Tests (No Changes to Existing)

Add new parallel versions alongside existing tests:
```cpp
// Keep existing test unchanged
TEST(AllReduce, OutOfPlace)
{
    // Original sequential implementation
}

// Add parallel version
TEST(AllReduce, OutOfPlace_Parallel)
{
    ParallelTestRunner runner;
    // Parallel implementation
    runner.executeAll();
}
```

Run both for validation:
```bash
# Run both versions
./rccl-UnitTests --gtest_filter="AllReduce.OutOfPlace*"
```

### Phase 2: Switch Default Execution

Once validated, replace sequential tests:
```cpp
TEST(AllReduce, OutOfPlace)
{
    // Check if parallel execution is enabled
    const char* parallelEnv = std::getenv("UT_PARALLEL_TESTS");
    if (parallelEnv && std::atoi(parallelEnv) != 0)
    {
        // Use parallel runner
        ParallelTestRunner runner;
        // ... setup ...
        runner.executeAll();
    }
    else
    {
        // Fall back to sequential
        TestBed testBed;
        // ... original implementation ...
    }
}
```

### Phase 3: Default to Parallel

Make parallel execution the default, with opt-out:
```bash
# Makefile or CI script
export UT_PARALLEL_TESTS=1  # Default

# Override for debugging
UT_PARALLEL_TESTS=0 ./rccl-UnitTests
```

## CI/CD Integration

### GitHub Actions / GitLab CI

```yaml
test:
  script:
    - export UT_PARALLEL_TESTS=1
    - export UT_MAX_PARALLEL_TESTS=8
    - ./rccl-UnitTests
  parallel:
    matrix:
      - TEST_SUITE: [AllReduce, Broadcast, ReduceScatter]
```

### Jenkins Pipeline

```groovy
stage('Test') {
    environment {
        UT_PARALLEL_TESTS = '1'
        UT_MAX_PARALLEL_TESTS = '8'
    }
    steps {
        sh './rccl-UnitTests'
    }
}
```

## Performance Metrics

Track these metrics to validate improvement:

```bash
# Before (baseline)
time UT_PARALLEL_TESTS=0 ./rccl-UnitTests > baseline.log 2>&1

# After (parallel)
time UT_PARALLEL_TESTS=1 ./rccl-UnitTests > parallel.log 2>&1

# Extract statistics
grep "Average GPU utilization" parallel.log
grep "Total execution time" parallel.log

# Compare wall clock time
echo "Baseline: $(grep real baseline.log)"
echo "Parallel: $(grep real parallel.log)"
```

## Implementation Files

```
test/
├── common/
│   ├── GPUScheduler.hpp              # Core scheduling engine
│   ├── GPUScheduler.cpp
│   ├── ParallelTestRunner.hpp        # High-level API
│   ├── ParallelTestRunner.cpp
│   └── PARALLEL_TESTING_GUIDE.md     # Detailed documentation
├── ParallelExecutionExample.cpp       # Example tests
└── PARALLEL_TESTING_README.md         # This file
```

## Next Steps

1. **Build and test**:
   ```bash
   cd build
   cmake .. -DBUILD_TESTS=ON
   make -j$(nproc)
   cd test
   ```

2. **Run examples**:
   ```bash
   export UT_PARALLEL_TESTS=1
   export UT_PARALLEL_VERBOSE=1
   ./rccl-UnitTests --gtest_filter="ParallelExecution.*"
   ```

3. **Measure improvement**:
   ```bash
   # Sequential
   time UT_PARALLEL_TESTS=0 ./rccl-UnitTests --gtest_filter="ParallelExecution.BasicSweep"

   # Parallel
   time UT_PARALLEL_TESTS=1 ./rccl-UnitTests --gtest_filter="ParallelExecution.BasicSweep"
   ```

4. **Integrate with existing tests**:
   - Start with one test suite (e.g., AllReduceTests.cpp)
   - Add parallel version
   - Validate results match
   - Expand to other test suites

## Contributing

When adding new tests, prefer the parallel-aware approach:

```cpp
TEST(NewFeature, MyTest)
{
    ParallelTestRunner runner;
    TestSweepBuilder builder(runner);

    builder.registerSweep("NewFeature_MyTest", 1, 8, [](int numGPUs) {
        return [numGPUs]() {
            // Test implementation
        };
    });

    ASSERT_TRUE(runner.executeAll());
}
```

This ensures your tests benefit from parallel execution automatically.

## Support

For issues or questions:
1. Check `PARALLEL_TESTING_GUIDE.md` for detailed documentation
2. Enable `UT_PARALLEL_VERBOSE=1` for debugging output
3. Compare with `ParallelExecutionExample.cpp` for reference implementations
