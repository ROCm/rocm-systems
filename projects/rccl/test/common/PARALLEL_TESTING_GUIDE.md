# GPU-Aware Parallel Test Execution Guide

## Overview

The GPU scheduler enables parallel execution of RCCL tests to maximize GPU utilization. Instead of running tests sequentially (1 GPU → 2 GPUs → ... → 8 GPUs), the scheduler runs multiple tests concurrently when their GPU requirements don't overlap.

## Quick Start

### Environment Variables

```bash
# Enable parallel test execution (default: 0)
export UT_PARALLEL_TESTS=1

# Maximum number of concurrent tests (default: 8)
export UT_MAX_PARALLEL_TESTS=8

# Enable verbose scheduling output (default: 0)
export UT_PARALLEL_VERBOSE=1
```

### Usage in Test Files

#### Option 1: Using ParallelTestRunner (Recommended for New Tests)

```cpp
#include "ParallelTestRunner.hpp"

TEST(AllReduce, ParallelSweep)
{
    using namespace RcclUnitTesting;

    ParallelTestRunner runner;

    // Register tests for different GPU counts
    for (int numGPUs = 1; numGPUs <= 8; ++numGPUs)
    {
        std::string testName = "AllReduce_OutOfPlace_" + std::to_string(numGPUs) + "GPU";

        runner.registerTest(testName, numGPUs, [numGPUs]() {
            TestBed testBed;
            std::vector<ncclFunc_t>     funcTypes       = {ncclCollAllReduce};
            std::vector<ncclDataType_t> dataTypes       = {ncclFloat32};
            std::vector<ncclRedOp_t>    redOps          = {ncclSum};
            std::vector<int>            roots           = {0};
            std::vector<int>            numElements     = {393216};
            std::vector<bool>           inPlaceList     = {false};
            std::vector<bool>           managedMemList  = {false};
            std::vector<bool>           useHipGraphList = {false};

            // Override GPU sweep to use specific count
            testBed.ev.minGpus = numGPUs;
            testBed.ev.maxGpus = numGPUs;

            testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                                  inPlaceList, managedMemList, useHipGraphList);
        });
    }

    // Execute all tests with GPU-aware scheduling
    ASSERT_TRUE(runner.executeAll());
}
```

#### Option 2: Using TestSweepBuilder (Even Simpler)

```cpp
#include "ParallelTestRunner.hpp"

TEST(AllReduce, AutomaticSweep)
{
    using namespace RcclUnitTesting;

    ParallelTestRunner runner;
    TestSweepBuilder sweepBuilder(runner);

    // Automatically create tests for GPU counts 1-8
    sweepBuilder.registerSweep("AllReduce_OutOfPlace", 1, 8,
        [](int numGPUs) {
            return [numGPUs]() {
                TestBed testBed;
                // ... configure test ...
                testBed.ev.minGpus = numGPUs;
                testBed.ev.maxGpus = numGPUs;
                testBed.RunSimpleSweep(/* ... */);
            };
        });

    ASSERT_TRUE(runner.executeAll());
}
```

#### Option 3: Backward Compatible (Keep Existing Tests)

Existing tests continue to work without modification. Simply set `UT_PARALLEL_TESTS=0` or don't set it at all.

## Performance Benefits

### Example Scenario
With 8 GPUs and tests requiring [1, 1, 2, 2, 4, 4, 8, 8] GPUs:

**Sequential Execution:**
```
Time 0-10:  1 GPU test  (7 GPUs idle)
Time 10-20: 1 GPU test  (7 GPUs idle)
Time 20-30: 2 GPU test  (6 GPUs idle)
Time 30-40: 2 GPU test  (6 GPUs idle)
Time 40-50: 4 GPU test  (4 GPUs idle)
Time 50-60: 4 GPU test  (4 GPUs idle)
Time 60-70: 8 GPU test  (0 GPUs idle)
Time 70-80: 8 GPU test  (0 GPUs idle)
Total: 80 time units
Average GPU utilization: ~35%
```

**Parallel Execution with GPU Scheduler:**
```
Time 0-10:  8 GPU test (GPU 0-7) + all GPUs busy
Time 10-20: 8 GPU test (GPU 0-7) + all GPUs busy
Time 20-30: 4 GPU test (GPU 0-3) + 4 GPU test (GPU 4-7) + all GPUs busy
Time 30-40: 4 GPU test (GPU 0-3) + 4 GPU test (GPU 4-7) + all GPUs busy
Time 40-50: 2 GPU test (GPU 0-1) + 2 GPU test (GPU 2-3) + 1 GPU test (GPU 4) + 1 GPU test (GPU 5)
Total: 50 time units
Average GPU utilization: ~85%
Speedup: 1.6x
```

## Advanced Usage

### Custom Priority

By default, tests requiring more GPUs have higher priority (scheduled first). You can override:

```cpp
runner.registerTest("CriticalTest", 2, testFunc,
                   999);  // Very high priority - runs first
```

### GPU Assignment Awareness

If your test needs to know which physical GPUs it's using:

```cpp
runner.registerTestWithGPUInfo("CustomTest", 4,
    [](const std::vector<int>& assignedGPUs) {
        // assignedGPUs might be [0,1,2,3] or [4,5,6,7] etc.
        INFO("Running on GPUs: ");
        for (int gpu : assignedGPUs) {
            INFO("%d ", gpu);
        }
        INFO("\n");

        // The environment variables are already set:
        // HIP_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES, ROCR_VISIBLE_DEVICES

        // Run your test...
    });
```

### Runtime Configuration

```cpp
ParallelTestRunner runner;

// Disable parallel execution for debugging
runner.setParallelExecution(false);

// Limit concurrent tests (useful for memory-constrained systems)
runner.setMaxConcurrentTests(4);

// Enable detailed logging
runner.setVerboseLogging(true);
```

## Migration Guide

### Converting Existing Tests

**Before:**
```cpp
TEST(AllReduce, OutOfPlace)
{
    TestBed testBed;
    // ... configuration ...
    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                          inPlaceList, managedMemList, useHipGraphList);
}
```

**After (with parallel execution):**
```cpp
TEST(AllReduce, OutOfPlace_Parallel)
{
    ParallelTestRunner runner;
    TestSweepBuilder sweep(runner);

    sweep.registerSweep("AllReduce_OutOfPlace",
                       1,  // min GPUs
                       8,  // max GPUs
                       [](int numGPUs) {
                           return [numGPUs]() {
                               TestBed testBed;
                               testBed.ev.minGpus = numGPUs;
                               testBed.ev.maxGpus = numGPUs;
                               // ... rest of configuration ...
                               testBed.RunSimpleSweep(/* ... */);
                           };
                       });

    ASSERT_TRUE(runner.executeAll());
}
```

**Keep Original (for comparison/validation):**
```cpp
// Keep the original test for validation
TEST(AllReduce, OutOfPlace_Sequential)
{
    // Original implementation unchanged
}
```

## Troubleshooting

### Tests fail only when running in parallel

This might indicate:
1. **Race conditions**: Tests are not properly isolated
2. **Shared resources**: Tests are inadvertently sharing GPU state
3. **Environment variable conflicts**: Check HIP_VISIBLE_DEVICES propagation

**Debug with:**
```bash
export UT_PARALLEL_TESTS=0  # Disable parallelism
export UT_PARALLEL_VERBOSE=1 # Enable detailed logging
```

### Out of memory errors

Reduce concurrent tests:
```bash
export UT_MAX_PARALLEL_TESTS=4
```

### GPU assignments not working

Verify environment variables are set correctly. The scheduler automatically sets:
- `HIP_VISIBLE_DEVICES`
- `CUDA_VISIBLE_DEVICES`
- `ROCR_VISIBLE_DEVICES`

## Implementation Details

### Architecture

```
ParallelTestRunner
    ├── Registers test jobs
    ├── Creates GPUScheduler
    └── Executes tests

GPUScheduler
    ├── Tracks GPU availability
    ├── Maintains job queue (priority-based)
    ├── Launches tests via fork()
    ├── Sets environment variables per child
    └── Monitors completion and releases GPUs
```

### Scheduling Algorithm

1. **Priority Queue**: Tests sorted by priority (default: larger GPU requirements = higher priority)
2. **Bin-Packing**: Larger tests scheduled first to minimize fragmentation
3. **Greedy Allocation**: As soon as resources are available, next test launches
4. **No Preemption**: Running tests complete before GPUs are reassigned

### GPU Allocation Strategy

- Allocates lowest-numbered available GPUs
- Uses environment variables to restrict visibility
- No GPU sharing (one GPU = one test at a time)
- Sequential allocation (GPU 0, 1, 2, ... , 7)

## Performance Tips

1. **Group similar tests**: Tests with similar GPU requirements can be batched
2. **Set priorities wisely**: Critical/long tests should run first
3. **Monitor utilization**: Use `UT_PARALLEL_VERBOSE=1` to see GPU usage
4. **Tune concurrency**: Adjust `UT_MAX_PARALLEL_TESTS` based on system memory

## Example Output

```
Registered test: AllReduce_1GPU (GPUs: 1, Priority: 90)
Registered test: AllReduce_2GPU (GPUs: 2, Priority: 80)
Registered test: AllReduce_8GPU (GPUs: 8, Priority: 20)
Executing 8 tests with GPU-aware scheduling
Allocated GPUs: 0,1,2,3,4,5,6,7
Launching test: AllReduce_8GPU (Job ID: 2) on 8 GPUs
Test completed: AllReduce_8GPU (Duration: 5432 ms, Status: PASSED)
Released GPUs: 0,1,2,3,4,5,6,7
Allocated GPUs: 0,1
Launching test: AllReduce_2GPU (Job ID: 1) on 2 GPUs
...

=============== GPU Scheduling Statistics ===============
Total tests executed: 8
Tests passed: 8
Tests failed: 0
Total execution time: 12456 ms
Average GPU utilization: 78.3%
========================================================
```

## Future Enhancements

Potential improvements (not yet implemented):
- Cross-node GPU scheduling (multi-node testing)
- Dynamic priority adjustment based on test history
- GPU affinity hints for NUMA optimization
- Automatic retry on transient failures
- Resource quotas per test suite
