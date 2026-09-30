# Process Isolated Test Runner

A lightweight C++ testing framework for running Google Test cases in isolated processes with clean environment settings.

## Table of Contents
- [Overview](#overview)
- [Why Use Process Isolation?](#why-use-process-isolation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Parallel Execution](#parallel-execution)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

`ProcessIsolatedTestRunner` is a framework that executes tests in separate processes using `fork()+execv()`. This ensures complete isolation between tests, particularly useful when testing code with static variables, one-time initialization, or environment-dependent behavior.

**Key Features:**
- ✅ Process-based test isolation via `fork()+execv()` (each test re-execs the binary fresh)
- ✅ Per-test environment variable management
- ✅ Configurable timeouts
- ✅ Sequential or parallel execution with bounded concurrency
- ✅ GPU-aware parallel scheduling (non-overlapping GPU assignment across concurrent tests)
- ✅ Thread-safe test registration
- ✅ Detailed test result reporting in registration order

**Location:** `test/common/ProcessIsolatedTestRunner.hpp`

---

## Why use Process Isolation?

### Problem: Static Variable Pollution

Consider this RCCL code with static variables:

```cpp
void rcclSetP2pNetChunkSize(struct ncclComm* comm, int& chunkSize) {
  static int p2pNetChunkSize = RCCL_VALUE_UNSET;  // ← Static variable!

  if (p2pNetChunkSize == RCCL_VALUE_UNSET) {
    const char* inputStr = getenv("NCCL_P2P_NET_CHUNKSIZE");
    if (inputStr) {
      // Parse the environment variable value
      p2pNetChunkSize = parseValue(inputStr);  // e.g., "12345" → 12345
    } else {
      // No env var set, calculate value based on architecture...
      p2pNetChunkSize = calculateValue();
    }
  }
  chunkSize = p2pNetChunkSize;
}
```

**How the static variable gets set:**
1. First time called: `p2pNetChunkSize == RCCL_VALUE_UNSET` is true
2. Code reads environment variable with `getenv("NCCL_P2P_NET_CHUNKSIZE")`
3. If env var exists → parse its value (e.g., "12345" string) and assign to static variable
4. If env var doesn't exist → calculate default value and assign to static variable
5. Static variable is now set and **persists for the lifetime of the process**

**Without Process Isolation:**
```cpp
TEST(MyTest, FirstTest) {
  setenv("NCCL_P2P_NET_CHUNKSIZE", "12345", 1);
  rcclSetP2pNetChunkSize(comm, chunkSize);
  // ✓ getenv() returns "12345"
  // ✓ Static variable p2pNetChunkSize gets set to 12345
  // ✓ chunkSize is now 12345
}

TEST(MyTest, SecondTest) {
  unsetenv("NCCL_P2P_NET_CHUNKSIZE");
  rcclSetP2pNetChunkSize(comm, chunkSize);
  // ❌ getenv() returns nullptr (env var cleared)
  // ❌ BUT: p2pNetChunkSize != RCCL_VALUE_UNSET (still 12345 from FirstTest!)
  // ❌ Code skips the if-block, never reads env var or recalculates
  // ❌ chunkSize is STILL 12345 from previous test!
  // This test will fail or produce incorrect results
}
```

**The Problem:** Static variables are initialized once per process and persist across multiple tests. Even if you change or clear environment variables, the static variable retains its old value.

**With Process Isolation:**
```cpp
// Each test runs in a separate process
// Static variables are reset for each test
// ✅ Tests are truly independent
```

### Common Use Cases

1. **Testing environment variable behavior** - When code reads env vars into static variables
2. **Testing architecture-specific logic** - Different GPU architectures with cached state
3. **Testing initialization code** - One-time initialization patterns
4. **Testing configuration changes** - When config is cached statically

---

## Quick Start

### Basic Example (Using Macros)

The simplest way to use ProcessIsolatedTestRunner is with the macros:

```cpp
#include "common/ProcessIsolatedTestRunner.hpp"

TEST(Rcclwrap, MyIsolatedTest) {
  // Single test with environment variables - all in one call!
  RUN_ISOLATED_TEST_WITH_ENV("TestWithCleanEnvironment",
    []() {
      // This runs in a separate process
      const char* value = getenv("MY_VARIABLE");
      EXPECT_STREQ(value, "test_value");
      EXPECT_TRUE(someFunction());
    },
    {{"MY_VARIABLE", "test_value"}}
  );
}

TEST(Rcclwrap, MyIsolatedTests) {
  // Multiple tests with different configurations
  RUN_ISOLATED_TESTS(
    ProcessIsolatedTestRunner::TestConfig("Test1", []() {
      EXPECT_TRUE(checkCondition1());
    }),
    ProcessIsolatedTestRunner::TestConfig("Test2", []() {
      EXPECT_TRUE(checkCondition2());
    }).withEnvironment({{"VAR", "value"}}),
    ProcessIsolatedTestRunner::TestConfig("Test3", []() {
      EXPECT_TRUE(checkCondition3());
    }).withTimeout(std::chrono::seconds(60))
  );
}
```

### Manual API (For Advanced Use Cases)

You can also use the API directly for more control:

```cpp
#include "common/ProcessIsolatedTestRunner.hpp"

TEST(Rcclwrap, MyIsolatedTests) {
  // Register a test with environment variables
  ProcessIsolatedTestRunner::registerTest(
      ProcessIsolatedTestRunner::TestConfig(
          "TestWithCleanEnvironment",
          []() {
            // This runs in a separate process
            const char* value = getenv("MY_VARIABLE");
            EXPECT_STREQ(value, "test_value");

            // Your test logic here
            EXPECT_TRUE(someFunction());
          })
          .withEnvironment({{"MY_VARIABLE", "test_value"}})
  );

  // Execute all registered tests
  bool allTestsPassed = ProcessIsolatedTestRunner::executeAllTests();
  EXPECT_TRUE(allTestsPassed);
}
```

---

## Core Concepts

### 1. Test Configuration (`TestConfig`)

Defines how a test should be executed:

```cpp
TestConfig config(
    "TestName",           // Test name (for reporting)
    []() { /* logic */ }  // Test function (lambda or function pointer)
);

// Optional configurations
config.withEnvironment({{"VAR1", "value1"}, {"VAR2", "value2"}})
      .withTimeout(std::chrono::seconds(60))
      .withCleanEnvironment(false);  // Inherit parent environment
```

### 2. Test Registration

Tests must be registered before execution:

```cpp
// Method 1: Full configuration
ProcessIsolatedTestRunner::registerTest(config);

// Method 2: Simple (name + logic only)
ProcessIsolatedTestRunner::registerTest("SimplTest", []() {
  EXPECT_TRUE(true);
});

// Method 3: With environment
ProcessIsolatedTestRunner::registerTest(
    "EnvTest",
    []() { /* logic */ },
    {{"ENV_VAR", "value"}}
);
```

### 3. Test Execution

**⚠️ IMPORTANT:** Tests do NOT run automatically after registration. You **MUST** explicitly call `executeAllTests()` to run them.

Execute all registered tests:

```cpp
// Default options (continue on failure, no verbose logging)
bool passed = ProcessIsolatedTestRunner::executeAllTests();

// Custom options
ProcessIsolatedTestRunner::ExecutionOptions options;
options.stopOnFirstFailure = true;   // Stop after first failure
options.verboseLogging = true;       // Print detailed logs

bool passed = ProcessIsolatedTestRunner::executeAllTests(options);
```

**Common Mistake:**
```cpp
// ❌ BAD: Tests registered but never executed!
TEST(MyTest, IsolatedTests) {
  ProcessIsolatedTestRunner::registerTest("Test1", []() { /* ... */ });
  ProcessIsolatedTestRunner::registerTest("Test2", []() { /* ... */ });
  // Missing executeAllTests() - tests will NOT run!
}

// ✅ GOOD: Tests registered and executed
TEST(MyTest, IsolatedTests) {
  ProcessIsolatedTestRunner::registerTest("Test1", []() { /* ... */ });
  ProcessIsolatedTestRunner::registerTest("Test2", []() { /* ... */ });
  bool passed = ProcessIsolatedTestRunner::executeAllTests();
  EXPECT_TRUE(passed);
}
```

### 4. Test Results

Each test produces a `TestResult`:

```cpp
struct TestResult {
  std::string testName;               // Name of the test
  bool passed;                        // Whether the test passed
  bool skipped;                       // Whether the test was skipped
  int exitCode;                       // Process exit code
  pid_t processId;                    // Process ID that ran the test
  std::chrono::milliseconds duration; // Execution duration
  std::string errorMessage;           // Error message if failed
  std::unordered_map<std::string, std::string> environment;  // Env used
};
```

---

## Parallel Execution

By default, tests run one at a time (`maxParallelJobs = 1`). Setting `maxParallelJobs > 1` launches up to that many child processes simultaneously, reducing total wall-clock time for large test suites.

### Concurrency control

```cpp
ProcessIsolatedTestRunner::ExecutionOptions opts;
opts.maxParallelJobs = 4;   // up to 4 tests run at the same time
                            // kAutoParallelism = GPU pool size (or hardware_concurrency() if no pool)
                            // 1 = sequential (default)

RUN_ISOLATED_TESTS_WITH_OPTIONS(opts,
    ProcessIsolatedTestRunner::TestConfig("Test1", []() { ... }),
    ProcessIsolatedTestRunner::TestConfig("Test2", []() { ... }),
    ProcessIsolatedTestRunner::TestConfig("Test3", []() { ... }),
    ProcessIsolatedTestRunner::TestConfig("Test4", []() { ... })
);
```

**Output ordering:** Results are always printed and reported in registration order, regardless of which child process finishes first.

**`stopOnFirstFailure`:** When set, the runner stops *launching* new tests once a failure is detected. Tests that are already running are allowed to finish normally — they are not killed early.

### GPU-aware scheduling

When tests run in parallel, multiple child processes would otherwise all try to use the same GPU (e.g., device 0), conflicting with each other. The runner solves this by maintaining a **GPU slot pool** and assigning each child process a non-overlapping subset of physical device indices, injected as `HIP_VISIBLE_DEVICES` before `execv()`.

#### How the pool is built

`ExecutionOptions::gpuPool` holds the physical device indices to distribute. If left empty (the default), the runner auto-detects the pool in this priority order:

1. Parse `HIP_VISIBLE_DEVICES` from the environment (comma-separated indices)
2. Count GPU nodes in `/sys/class/kfd/kfd/topology/nodes` with a non-zero `gpu_id` → pool `[0, 1, ..., N-1]`

If neither source yields a non-empty pool, `executeAllTests()` records a non-fatal GTest failure and returns `false` immediately.

```cpp
// Override auto-detection to use only GPUs 0 and 1:
opts.gpuPool = {0, 1};
```

#### How slots are assigned

Each `TestConfig` declares how many GPU slots it needs via `withNumGpus()`:

| `numGpus` value | Meaning |
|---|---|
| `0` (default) | CPU-only test — no GPU slot acquired, runs freely without `HIP_VISIBLE_DEVICES` restriction |
| `1` | one dedicated GPU from the pool |
| `N` | exactly N GPU slots — test blocks until N slots are free |
| `N > pool size` | clamped to the full pool — test runs exclusively (all slots held) to prevent contention with siblings |

Before launching each child, the runner atomically waits until **both** a concurrency slot (`active < maxParallelJobs`) **and** enough free GPU slots are available. The assigned indices are written into `HIP_VISIBLE_DEVICES` inside the fork child so the re-exec'd process sees only its GPUs. When the child exits the slots are released and the next waiting test can proceed.

#### Example: parallel tests with per-test GPU declaration

```cpp
ProcessIsolatedTestRunner::ExecutionOptions opts;
opts.maxParallelJobs = 4;
// gpuPool auto-detected (e.g. [0,1,2,3] on a 4-GPU node)

RUN_ISOLATED_TESTS_WITH_OPTIONS(opts,
    // Each of these gets one GPU; up to 4 run simultaneously
    ProcessIsolatedTestRunner::TestConfig("SingleGpuTest_A", []() {
        // HIP_VISIBLE_DEVICES is set to e.g. "0" — test sees only that GPU
        HIPCALL(hipSetDevice(0));  // device 0 here = whichever physical GPU was assigned
        /* ... */
    }),
    ProcessIsolatedTestRunner::TestConfig("SingleGpuTest_B", []() {
        HIPCALL(hipSetDevice(0));
        /* ... */
    }),

    // This test needs 2 GPUs — it blocks until 2 slots are free
    ProcessIsolatedTestRunner::TestConfig("TwoGpuTest", []() {
        // HIP_VISIBLE_DEVICES = e.g. "2,3"
        // hipGetDeviceCount() returns 2; hipSetDevice(0/1) map to physical 2/3
        /* ... */
    }).withNumGpus(2),

    // This test requires all 4 GPUs — blocks until every slot is free,
    // so no other GPU test can run concurrently (exclusive execution).
    ProcessIsolatedTestRunner::TestConfig("ExclusiveTest", []() {
        /* ... */
    }).withNumGpus(4)   // request entire 4-GPU pool → runs exclusively
);
```

#### What the child process sees

Inside the re-exec'd child, `HIP_VISIBLE_DEVICES` is set to the assigned indices (e.g. `"2,3"`). HIP then remaps those to logical device IDs 0 and 1. So a test that calls `hipSetDevice(0)` always gets the first device from its assigned subset — it never touches a GPU that belongs to another concurrent test.

```
System GPUs:  [0]  [1]  [2]  [3]
              └─ Test A ─┘  └─ Test B ─┘   ← running simultaneously
  HIP_VISIBLE_DEVICES="0,1"  "2,3"
  hipSetDevice(0) → phys 0   hipSetDevice(0) → phys 2
```

#### When GPU slot management is disabled

GPU slot management is skipped (no `HIP_VISIBLE_DEVICES` override) in two cases:
- **Sequential mode** (`maxParallelJobs = 1`): no concurrent tests, no conflict possible.
- **No GPUs detected**: auto-detection found zero devices.

In both cases tests see the full device list as usual.

#### Interaction with `withEnvironment({{"HIP_VISIBLE_DEVICES", ...}})`

In parallel mode, the runner injects `HIP_VISIBLE_DEVICES` **after** applying the test's own environment variables. Any `HIP_VISIBLE_DEVICES` set via `withEnvironment()` or `setVariable()` is overridden by the slot manager's assignment; a warning is logged to stderr when this happens. Use `withNumGpus(N)` to declare device requirements rather than hard-coding device indices.

---

## API Reference

### Macros (Recommended)

These macros provide the simplest way to use ProcessIsolatedTestRunner with minimal boilerplate.

#### `RUN_ISOLATED_TEST(test_name, test_body)`
Register and execute a single isolated test.

```cpp
RUN_ISOLATED_TEST("MySimpleTest", []() {
  EXPECT_TRUE(someFunction());
});
```

#### `RUN_ISOLATED_TEST_WITH_ENV(test_name, test_body, ...)`
Register and execute a single isolated test with environment variables.

**Uses variadic macros** (`...` and `__VA_ARGS__`) to automatically handle commas in initializer lists without requiring extra parentheses.

```cpp
RUN_ISOLATED_TEST_WITH_ENV("MyEnvTest",
  []() {
    const char* value = getenv("MY_VAR");
    EXPECT_STREQ(value, "expected_value");
  },
  {{"MY_VAR", "expected_value"}}
);

// Multiple environment variables work naturally:
RUN_ISOLATED_TEST_WITH_ENV("MultiEnvTest",
  []() { /* test code */ },
  {{"VAR1", "val1"}, {"VAR2", "val2"}, {"VAR3", "val3"}}  // Commas handled automatically
);
```

**Note:** The macro uses `__VA_ARGS__` internally, which automatically handles commas in the environment variable initializer list. Users don't need to worry about preprocessor comma issues.

#### `RUN_ISOLATED_TESTS(...)`
Register and execute multiple isolated tests with various configurations.

```cpp
RUN_ISOLATED_TESTS(
  ProcessIsolatedTestRunner::TestConfig("Test1", []() { ... }),
  ProcessIsolatedTestRunner::TestConfig("Test2", []() { ... })
    .withEnvironment({{"VAR", "value"}}),
  ProcessIsolatedTestRunner::TestConfig("Test3", []() { ... })
    .withTimeout(std::chrono::seconds(60))
);
```

#### `RUN_ISOLATED_TESTS_WITH_OPTIONS(options, ...)`
Register and execute multiple isolated tests with custom execution options.

```cpp
ProcessIsolatedTestRunner::ExecutionOptions opts;
opts.stopOnFirstFailure = true;
opts.verboseLogging = true;

RUN_ISOLATED_TESTS_WITH_OPTIONS(opts,
  ProcessIsolatedTestRunner::TestConfig("Test1", []() { ... }),
  ProcessIsolatedTestRunner::TestConfig("Test2", []() { ... })
);
```

### Main Methods (For Manual Use)

#### `registerTest()`
Register a test for later execution. Registration fails with a diagnostic if:
- The name contains a null byte (would be silently truncated by `setenv`/`getenv`).
- The name is a duplicate of an already-registered test (the re-exec child matches on the first occurrence, so the second would never run).

```cpp
// Variant 1: Full configuration
static void registerTest(const TestConfig& config);

// Variant 2: Simple registration
static void registerTest(
    const std::string& name,
    std::function<void()> testLogic
);

// Variant 3: With environment
static void registerTest(
    const std::string& name,
    std::function<void()> testLogic,
    const std::unordered_map<std::string, std::string>& env
);
```

#### `executeAllTests()`
Execute all registered tests.

```cpp
static bool executeAllTests(
    const ExecutionOptions& options = ExecutionOptions()
);
```

**Returns:** `true` if all tests passed, `false` if any failed.

**Execution mode** is controlled by `ExecutionOptions::maxParallelJobs`:
- `1` (default) — sequential, one test at a time
- `N > 1` — up to N tests run simultaneously with GPU-aware scheduling
- `ExecutionOptions::kAutoParallelism` (`0`) — parallelism = GPU pool size, or `std::thread::hardware_concurrency()` when no GPU pool is available

**Note:** This method automatically clears all test registrations and results after execution, ensuring a clean state for the next test suite. Users do not need to call `clear()` manually.

#### `getTestResults()`
Retrieve detailed results from the last execution.

```cpp
static std::vector<TestResult> getTestResults();
```

#### `clear()`
Clear all registered tests and results.

```cpp
static void clear();
```

**Note:** Calling this method manually is typically not necessary, as `executeAllTests()` automatically clears registrations after execution. This method is primarily useful for advanced use cases or when tests are registered but not executed.

**⚠️ Automatic Warning:** If `clear()` is called when tests have been registered but not fully executed, it will automatically print a warning to stderr:

```
⚠️  WARNING: ProcessIsolatedTestRunner::clear() called with 2 unexecuted test(s)!
   Registered: 2 test(s)
   Executed:   0 test(s)
   Did you forget to call executeAllTests()?
```

#### `getTestCount()`
Get the number of currently registered tests (before execution).

```cpp
static size_t getTestCount();
```

**Use case:** Verify that tests were actually registered and executed.

```cpp
TEST(MyTest, VerifyExecution) {
  ProcessIsolatedTestRunner::clear();

  // Register tests
  ProcessIsolatedTestRunner::registerTest("Test1", []() { /* ... */ });
  ProcessIsolatedTestRunner::registerTest("Test2", []() { /* ... */ });

  // Check registration count
  size_t registeredCount = ProcessIsolatedTestRunner::getTestCount();
  EXPECT_EQ(registeredCount, 2) << "Expected 2 tests to be registered";

  // Execute
  bool passed = ProcessIsolatedTestRunner::executeAllTests();
  EXPECT_TRUE(passed);

  // Verify execution count
  auto results = ProcessIsolatedTestRunner::getTestResults();
  EXPECT_EQ(results.size(), registeredCount)
      << "Registered " << registeredCount << " tests but only "
      << results.size() << " executed";
}
```

### ExecutionOptions Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `stopOnFirstFailure` | `bool` | `false` | Stop launching new tests after the first failure (in-flight tests finish) |
| `verboseLogging` | `bool` | `true` | Print per-test status lines |
| `maxParallelJobs` | `size_t` | `1` | Max concurrent child processes. `kAutoParallelism` (`0`) = GPU pool size (or `hardware_concurrency` if no pool), `1` = sequential |
| `gpuPool` | `vector<int>` | `{}` | Physical GPU indices to distribute. Empty = auto-detect |

### TestConfig Methods

#### `withEnvironment()`
Set environment variables for the test.

```cpp
TestConfig& withEnvironment(
    const std::unordered_map<std::string, std::string>& env
);
```

**Note:** Variables are set in the child process only.

#### `withTimeout()`
Set a timeout for test execution.

```cpp
TestConfig& withTimeout(std::chrono::seconds timeoutSeconds);
```

**Default:** 30 seconds

#### `withCleanEnvironment()`
Control whether to inherit parent process environment.

```cpp
TestConfig& withCleanEnvironment(bool inherit = true);
```

**Default:** `true` (inherits parent environment)

#### `withNumGpus()`
Declare how many GPU slots this test needs during parallel execution.

```cpp
TestConfig& withNumGpus(size_t n);
```

| Value | Meaning |
|---|---|
| `0` (default) | CPU-only — no GPU slot acquired; `HIP_VISIBLE_DEVICES` is not overridden |
| `1` | one dedicated GPU from the pool |
| `N` | exactly N GPU slots — test blocks until N slots are free |
| `N > pool size` | clamped to pool size — test runs exclusively (holds all slots) |

The runner injects `HIP_VISIBLE_DEVICES` into the child process with the assigned device indices. Has no effect in sequential mode (`maxParallelJobs = 1`).

```cpp
ProcessIsolatedTestRunner::TestConfig("MultiGpuTest", []() {
    // Sees exactly 2 GPUs via HIP_VISIBLE_DEVICES
    int count;
    hipGetDeviceCount(&count);
    EXPECT_EQ(count, 2);
}).withNumGpus(2)
```

---

## Examples

**Note:** The examples below use helper functions from `RcclWrapTests.cpp`:

```cpp
// Helper to create a mock NCCL communicator with specified architecture and ranks
static void CreateMockComm(ncclComm_t &mockComm,
                           struct ncclTopoSystem &mockTopo,
                           struct ncclTopoNode &mockGpuNode,
                           const char *arch,
                           int nRanks);

// Helper to cleanup a mock communicator
static void CleanupMockComm(ncclComm_t &mockComm);
```

### Example 1: Testing Environment Variable Behavior

```cpp
TEST(Rcclwrap, EnvironmentVariableTests) {
  // Test 1: With environment variable set
  ProcessIsolatedTestRunner::registerTest(
      ProcessIsolatedTestRunner::TestConfig(
          "WithEnvVarSet",
          []() {
            ncclComm_t mockComm = nullptr;
            struct ncclTopoSystem mockTopo;
            struct ncclTopoNode mockGpuNode;
            CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 128);

            int chunkSize = RCCL_VALUE_UNSET;
            rcclSetP2pNetChunkSize(mockComm, chunkSize);

            // Should use default architecture-based value
            EXPECT_EQ(chunkSize, 1 << 19);

            CleanupMockComm(mockComm);
          })
          .withEnvironment({{"NCCL_P2P_NET_CHUNKSIZE", "999999"}})
  );

  // Test 2: Without environment variable (clean state)
  ProcessIsolatedTestRunner::registerTest(
      ProcessIsolatedTestRunner::TestConfig(
          "WithoutEnvVar",
          []() {
            // Verify environment is clean
            const char* value = getenv("NCCL_P2P_NET_CHUNKSIZE");
            EXPECT_EQ(value, nullptr);

            // Test default behavior
            ncclComm_t mockComm = nullptr;
            struct ncclTopoSystem mockTopo;
            struct ncclTopoNode mockGpuNode;
            CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 32);

            int chunkSize = RCCL_VALUE_UNSET;
            rcclSetP2pNetChunkSize(mockComm, chunkSize);
            EXPECT_EQ(chunkSize, 1 << 17);  // Default for < 64 ranks

            CleanupMockComm(mockComm);
          })
  );

  // Execute both tests in isolated processes
  bool passed = ProcessIsolatedTestRunner::executeAllTests();
  EXPECT_TRUE(passed);
}
```

### Example 2: Testing Multiple Architectures

```cpp
TEST(Rcclwrap, ArchitectureTests) {
  struct TestCase {
    std::string name;
    std::string arch;
    int ranks;
    int expectedChunkSize;
  };

  std::vector<TestCase> testCases = {
    {"GFX942_SmallRanks", "gfx942", 32, 1 << 17},
    {"GFX942_LargeRanks", "gfx942", 128, 1 << 19},
    {"GFX950_SmallRanks", "gfx950", 8, 1 << 17},
    {"GFX950_MediumRanks", "gfx950", 24, 1 << 18},
    {"GFX950_LargeRanks", "gfx950", 64, 1 << 19},
  };

  for (const auto& tc : testCases) {
    ProcessIsolatedTestRunner::registerTest(
        ProcessIsolatedTestRunner::TestConfig(
            tc.name,
            [tc]() {
              ncclComm_t mockComm = nullptr;
              struct ncclTopoSystem mockTopo;
              struct ncclTopoNode mockGpuNode;
              CreateMockComm(mockComm, mockTopo, mockGpuNode, tc.arch.c_str(), tc.ranks);

              int chunkSize = RCCL_VALUE_UNSET;
              rcclSetP2pNetChunkSize(mockComm, chunkSize);

              EXPECT_EQ(chunkSize, tc.expectedChunkSize)
                  << "Failed for " << tc.arch << " with " << tc.ranks << " ranks";

              CleanupMockComm(mockComm);
            })
    );
  }

  ProcessIsolatedTestRunner::ExecutionOptions options;
  options.verboseLogging = true;
  options.stopOnFirstFailure = false;  // Run all tests even if one fails

  bool passed = ProcessIsolatedTestRunner::executeAllTests(options);
  EXPECT_TRUE(passed);
}
```

### Example 3: Testing with Timeouts

```cpp
TEST(Rcclwrap, TimeoutHandling) {
  // Test that completes quickly
  ProcessIsolatedTestRunner::registerTest(
      ProcessIsolatedTestRunner::TestConfig(
          "FastTest",
          []() {
            EXPECT_TRUE(true);
          })
          .withTimeout(std::chrono::seconds(5))
  );

  // Test with longer timeout for complex operations
  ProcessIsolatedTestRunner::registerTest(
      ProcessIsolatedTestRunner::TestConfig(
          "SlowTest",
          []() {
            // Simulate slow operation
            std::this_thread::sleep_for(std::chrono::seconds(2));
            EXPECT_TRUE(true);
          })
          .withTimeout(std::chrono::seconds(10))
  );

  bool passed = ProcessIsolatedTestRunner::executeAllTests();
  EXPECT_TRUE(passed);
}
```

### Example 4: Stop on First Failure

```cpp
TEST(Rcclwrap, CriticalTests) {
  // Register multiple critical tests
  ProcessIsolatedTestRunner::registerTest(
      "CriticalTest1", []() { EXPECT_TRUE(checkCriticalCondition1()); });

  ProcessIsolatedTestRunner::registerTest(
      "CriticalTest2", []() { EXPECT_TRUE(checkCriticalCondition2()); });

  ProcessIsolatedTestRunner::registerTest(
      "CriticalTest3", []() { EXPECT_TRUE(checkCriticalCondition3()); });

  // Stop on first failure - don't waste time if critical tests fail
  ProcessIsolatedTestRunner::ExecutionOptions options;
  options.stopOnFirstFailure = true;

  bool passed = ProcessIsolatedTestRunner::executeAllTests(options);
  EXPECT_TRUE(passed) << "Critical test suite failed";
}
```

---

## Best Practices

### 1. Use Macros for Simple Cases

```cpp
// ✅ GOOD: Simple and clean using macros
TEST(MyTest, SimpleIsolatedTest) {
  RUN_ISOLATED_TEST("CheckSomething", []() {
    EXPECT_TRUE(checkSomething());
  });
}

// ❌ MORE VERBOSE: Manual registration (still valid for complex cases)
TEST(MyTest, SimpleIsolatedTest) {
  ProcessIsolatedTestRunner::registerTest("CheckSomething", []() {
    EXPECT_TRUE(checkSomething());
  });
  bool passed = ProcessIsolatedTestRunner::executeAllTests();
  EXPECT_TRUE(passed);
}
```

### 2. Always Execute Registered Tests (When Using Manual API)

```cpp
TEST(MyTest, IsolatedTests) {
  // Register tests
  ProcessIsolatedTestRunner::registerTest(/* ... */);

  // ✅ IMPORTANT: Don't forget to execute!
  bool passed = ProcessIsolatedTestRunner::executeAllTests();
  EXPECT_TRUE(passed);
}
```

**When Using Manual API (Optional Verification):**

You can verify that tests were registered and executed:

```cpp
TEST(MyTest, IsolatedTests) {
  // Register tests
  ProcessIsolatedTestRunner::registerTest("Test1", []() { /* ... */ });
  ProcessIsolatedTestRunner::registerTest("Test2", []() { /* ... */ });

  // Get count of registered tests
  size_t registeredCount = ProcessIsolatedTestRunner::getTestCount();
  EXPECT_EQ(registeredCount, 2) << "Expected 2 tests to be registered";

  // Execute all tests (automatically clears after execution)
  bool passed = ProcessIsolatedTestRunner::executeAllTests();
  EXPECT_TRUE(passed);

  // Optional: Verify execution count matches registration count
  auto results = ProcessIsolatedTestRunner::getTestResults();
  EXPECT_EQ(results.size(), registeredCount)
      << "Registered " << registeredCount << " but executed " << results.size();
}
```

### 3. Use Descriptive Test Names

```cpp
// ❌ BAD: Vague name
RUN_ISOLATED_TEST("Test1", []() { /* ... */ });

// ✅ GOOD: Descriptive name
RUN_ISOLATED_TEST("GFX942_LargeRanks_P2PChunkSize_ExpectHighValue",
  []() { /* ... */ }
);
```

### 4. Group Related Tests

```cpp
TEST(Rcclwrap, AllP2PChunkSizeTests) {
  // Using macros to group related tests
  RUN_ISOLATED_TESTS(
    ProcessIsolatedTestRunner::TestConfig("GFX942_Test1", []() { ... }),
    ProcessIsolatedTestRunner::TestConfig("GFX942_Test2", []() { ... }),
    ProcessIsolatedTestRunner::TestConfig("GFX950_Test1", []() { ... }),
    ProcessIsolatedTestRunner::TestConfig("GFX950_Test2", []() { ... })
  );
}
```

### 5. Use Options for Better Control

```cpp
// For debugging: verbose + stop on failure
ProcessIsolatedTestRunner::ExecutionOptions debugOptions;
debugOptions.stopOnFirstFailure = true;
debugOptions.verboseLogging = true;

RUN_ISOLATED_TESTS_WITH_OPTIONS(debugOptions,
  ProcessIsolatedTestRunner::TestConfig("Test1", []() { ... }),
  ProcessIsolatedTestRunner::TestConfig("Test2", []() { ... })
);

// For CI: run all tests, collect all failures
ProcessIsolatedTestRunner::ExecutionOptions ciOptions;
ciOptions.stopOnFirstFailure = false;
ciOptions.verboseLogging = false;

RUN_ISOLATED_TESTS_WITH_OPTIONS(ciOptions,
  ProcessIsolatedTestRunner::TestConfig("Test1", []() { ... }),
  ProcessIsolatedTestRunner::TestConfig("Test2", []() { ... })
);
```

### 6. Set Appropriate Timeouts

```cpp
// ✅ GOOD: Different timeouts for different test types
RUN_ISOLATED_TESTS(
  ProcessIsolatedTestRunner::TestConfig("QuickTest", []() { ... })
    .withTimeout(std::chrono::seconds(5)),
  ProcessIsolatedTestRunner::TestConfig("NormalTest", []() { ... })
    .withTimeout(std::chrono::seconds(30)),
  ProcessIsolatedTestRunner::TestConfig("SlowTest", []() { ... })
    .withTimeout(std::chrono::seconds(120))
);

// ❌ BAD: Same long timeout for everything
RUN_ISOLATED_TESTS(
  ProcessIsolatedTestRunner::TestConfig("Test1", []() { ... })
    .withTimeout(std::chrono::seconds(300)),
  ProcessIsolatedTestRunner::TestConfig("Test2", []() { ... })
    .withTimeout(std::chrono::seconds(300))
);
```

### 7. Clean Up Resources in Tests

```cpp
RUN_ISOLATED_TEST("ResourceTest", []() {
  ncclComm_t comm = nullptr;
  struct ncclTopoSystem topo;
  struct ncclTopoNode gpuNode;
  CreateMockComm(comm, topo, gpuNode, "gfx942", 32);

  try {
    // Your test logic
    EXPECT_TRUE(someTest(comm));

    // ✅ GOOD: Clean up in all paths
    CleanupMockComm(comm);
  } catch (...) {
    CleanupMockComm(comm);
    throw;
  }
});
```

### 8. Use RAII for GPU Resource Management

When tests allocate GPU memory, use RAII wrappers to ensure cleanup:

```cpp
// ✅ GOOD: RAII ensures cleanup even on failure
struct GPUBuffer {
  void* ptr = nullptr;
  size_t size;

  GPUBuffer(size_t s) : size(s) {
    hipError_t err = hipMalloc(&ptr, size);
    ASSERT_EQ(err, hipSuccess);
  }

  ~GPUBuffer() {
    if (ptr) {
      hipFree(ptr);
      ptr = nullptr;
    }
  }

  // Prevent copying
  GPUBuffer(const GPUBuffer&) = delete;
  GPUBuffer& operator=(const GPUBuffer&) = delete;
};

RUN_ISOLATED_TEST("GPUTest", []() {
  GPUBuffer buffer(1024);  // Automatically cleaned up
  // ... test logic ...
  // No manual cleanup needed - destructor handles it
});

// ❌ BAD: Manual cleanup can be forgotten
RUN_ISOLATED_TEST("GPUTest", []() {
  void* buffer;
  hipMalloc(&buffer, 1024);
  // ... test logic ...
  // If test fails before this line, buffer leaks!
  hipFree(buffer);
});
```

### 9. Avoid GPU Initialization in Test Fixtures

When using process isolation, avoid initializing GPU resources in test fixture `SetUp()` methods:

```cpp
// ❌ BAD: GPU initialization in fixture (runs in parent process)
class GPUTests : public ::testing::Test {
protected:
  void SetUp() override {
    hipMalloc(&gpuBuffer, 1024);  // Parent process — unsafe; HIP state
                                  // must not be inherited across fork()+execv()
  }
  void* gpuBuffer;
};

// ✅ GOOD: GPU initialization inside isolated test
class GPUTests : public ::testing::Test {
  // Empty fixture or only CPU resources in SetUp()
};

TEST_F(GPUTests, MyTest) {
  RUN_ISOLATED_TEST("GPUOperation", []() {
    void* gpuBuffer;
    hipMalloc(&gpuBuffer, 1024);  // Child process only - safe!
    // ... test logic ...
    hipFree(gpuBuffer);
  });
}

// ✅ EVEN BETTER: Use RAII + helper structure
struct GPUTestEnvironment {
  void* buffer;
  void setup() { hipMalloc(&buffer, 1024); }
  void cleanup() { if (buffer) hipFree(buffer); }
  ~GPUTestEnvironment() { cleanup(); }
};

TEST_F(GPUTests, MyTest) {
  RUN_ISOLATED_TEST("GPUOperation", []() {
    GPUTestEnvironment env;
    env.setup();
    // ... test logic ...
    env.cleanup();  // Explicit + destructor cleanup
  });
}
```

---

## Troubleshooting

### Test Hangs / Times Out

**Symptom:** Test never completes, eventually times out.

**Solutions:**
1. Increase timeout: `.withTimeout(std::chrono::seconds(120))`
2. Check for deadlocks in test logic
3. Enable verbose logging to see where it hangs:
   ```cpp
   options.verboseLogging = true;
   ```

### Environment Variables Not Being Set

**Symptom:** `getenv()` returns `nullptr` in test.

**Solutions:**
1. Verify environment variable name is correct
2. Check that you're calling `withEnvironment()`:
   ```cpp
   config.withEnvironment({{"VAR_NAME", "value"}})
   ```
3. Verify the test is actually executing (check test name)

### Tests Pass Individually but Fail Together

**Symptom:** Individual tests pass, but fail when run in a suite.

**Cause:** This is the **exact problem** that ProcessIsolatedTestRunner solves!

**Solution:** Already solved - each test runs in isolated process. If you're still seeing this, check:
1. Are you using `executeAllTests()` correctly?
2. Are there shared external resources (files, network, etc.)?

### Fork Failures

**Symptom:** Error messages about fork() failing.

**Solutions:**
1. Check system resource limits: `ulimit -u` (max processes)
2. Reduce number of tests or run in smaller batches
3. Check for resource leaks in parent process

### Test Results Not Available

**Symptom:** `getTestResults()` returns empty vector.

**Solution:**
```cpp
// Call executeAllTests() first
ProcessIsolatedTestRunner::executeAllTests();

// Then get results
auto results = ProcessIsolatedTestRunner::getTestResults();
```

### Tests Registered but Never Executed

**Symptom:** Tests pass but you suspect they didn't actually run.

**Cause:** Forgot to call `executeAllTests()` after registration.

**Detection:**
```cpp
TEST(MyTest, IsolatedTests) {
  // Register tests
  ProcessIsolatedTestRunner::registerTest("Test1", []() { EXPECT_TRUE(true); });
  ProcessIsolatedTestRunner::registerTest("Test2", []() { EXPECT_TRUE(true); });

  // ❌ FORGOT TO CALL executeAllTests()!

  // Later, when the test ends, registered tests are lost
}
```

**Solution:**
```cpp
TEST(MyTest, IsolatedTests) {
  // Register tests
  ProcessIsolatedTestRunner::registerTest("Test1", []() { EXPECT_TRUE(true); });
  ProcessIsolatedTestRunner::registerTest("Test2", []() { EXPECT_TRUE(true); });

  // ✅ ALWAYS execute registered tests
  bool passed = ProcessIsolatedTestRunner::executeAllTests();
  EXPECT_TRUE(passed);

  // ✅ Optionally verify execution count
  auto results = ProcessIsolatedTestRunner::getTestResults();
  EXPECT_EQ(results.size(), 2) << "Expected 2 tests to execute";
}
```

**Prevention:** Always verify that `getTestResults().size()` matches your expected number of tests:
```cpp
// After execution
auto results = ProcessIsolatedTestRunner::getTestResults();
EXPECT_EQ(results.size(), expectedTestCount)
    << "Test count mismatch - some tests may not have executed";
```

---

## Implementation Details

### How It Works

1. **Registration Phase:**
   - Tests are registered into a static vector
   - Each test gets a `TestConfig` with name, logic, and environment

2. **Execution Phase (exec-only isolation):**
   - Parent detects the GPU pool once (from `HIP_VISIBLE_DEVICES` or KFD sysfs); fails with a non-fatal GTest error and returns `false` if the pool is empty
   - For each test the parent:
     1. Acquires a concurrency slot (`active < maxParallelJobs`) and GPU slots (`numGpus` free indices) — atomically under one condition variable
     2. `fork()` creates a child
     3. Inside the fork child: environment variables are applied, then `HIP_VISIBLE_DEVICES` is set to the assigned GPU indices, then `execv("/proc/self/exe")` re-executes the binary with `--gtest_filter` pointing at the parent test and a sentinel env var naming the specific isolated test to run
     4. The re-exec'd process finds the sentinel, runs the matching lambda, flushes coverage, and calls `_exit()`
     5. Parent thread drains stdout/stderr pipes and calls `waitpid()` for its child
     6. Concurrency and GPU slots are released; the next waiting test can start
   - In parallel mode, each test runs in a background `std::async` thread; the parent maintains a bounded sliding window of `maxParallelJobs` active tasks

3. **Result Collection:**
   - Exit codes are captured from child processes
   - Timing information is recorded
   - All results are stored and reported in registration order

4. **Automatic Cleanup:**
   - After execution completes, `executeAllTests()` automatically clears all test registrations and results
   - This ensures a clean state for the next test suite without manual intervention

### Why `fork()+execv()` instead of plain `fork()`

Plain `fork()` after HIP has been initialized is unsafe — the GPU runtime state (device contexts, memory handles, threads) is copied into the child but is not valid there. `execv()` replaces the child's entire address space with a fresh binary image, so the child starts with no inherited runtime state. This also ensures the LLVM profile runtime initializes with the correct child PID. Before `execv()` the fork child sets `LLVM_PROFILE_FILE=rccl_tests_%p_%m.profraw` (with `overwrite=0`) so each re-exec'd process writes a unique file. A `LLVM_PROFILE_FILE` already in the environment — from CI or via `withEnvironment()` — takes precedence.

The re-exec'd child skips GPU enumeration calls in `EnvVars` (e.g., `hipGetDeviceCount`, `getArchInfo`) because they are irrelevant there and concurrent `hipGetDeviceCount` forks cause KFD file-descriptor contention.

### Exit Codes

```cpp
enum RcclTestCode {
  RCCL_TEST_SUCCESS = 0,           // Test passed
  RCCL_TEST_FAILURE = 1,           // Test failed (assertion)
  RCCL_TEST_UNKNOWN_EXCEPTION = 2, // Uncaught exception
  RCCL_TEST_TIMEOUT = 3,           // Test timed out
  RCCL_TEST_SKIPPED = 4            // Test was skipped
};
```

### Thread Safety

The framework uses mutexes for thread-safe operations:
- Test registration (write)
- Result recording (write)
- Result retrieval (read)

---

## Limitations

1. **Process Overhead:** Each test forks and re-execs the binary (higher overhead than plain `fork()`)
2. **Linux/Unix Only:** Uses `fork()+execv()` — not available on Windows
3. **No Shared State:** Tests cannot share data between processes
4. **GPU pool is fixed at launch:** The GPU pool is detected once before the first test runs; hot-plug changes during a run are not reflected

---

## FAQ

**Q: When should I use ProcessIsolatedTestRunner vs regular Google Test?**

A: Use ProcessIsolatedTestRunner when:
- Testing code with static variables
- Testing environment variable behavior
- Testing one-time initialization
- Need guaranteed clean state between tests

Use regular Google Test when:
- Tests are truly independent
- No static state concerns
- Need parallel execution
- Testing simple units

**Q: Can I use this with MPI tests?**

A: Not directly. Process Isolated test runner is for single-process tests. For MPI tests, use `MPI Test Runner` instead. Process Isolated test runner is currently hooked into `rccl-UnitTestsFixtures` binary and MPI test runner is hooked into `rccl-UnitTestsMPI` binary. These are two independent implementation.

**Q: How do I debug a test that's running in an isolated process?**

A:
1. Enable verbose logging
2. Add print statements in your test lambda
3. Temporarily run the test logic outside the framework
4. Use GDB

**Q: Can I run tests in parallel?**

A: Yes. Set `ExecutionOptions::maxParallelJobs` to a value greater than 1. The runner spawns up to that many child processes simultaneously and automatically assigns non-overlapping GPU subsets to each so concurrent tests do not interfere on the GPU. Results are always reported in registration order.

```cpp
ProcessIsolatedTestRunner::ExecutionOptions opts;
opts.maxParallelJobs = 4;  // 4 concurrent tests

RUN_ISOLATED_TESTS_WITH_OPTIONS(opts,
    ProcessIsolatedTestRunner::TestConfig("A", []() { ... }),
    ProcessIsolatedTestRunner::TestConfig("B", []() { ... }),
    ProcessIsolatedTestRunner::TestConfig("C", []() { ... }),
    ProcessIsolatedTestRunner::TestConfig("D", []() { ... })
);
```

See the [Parallel Execution](#parallel-execution) section for full details on GPU slot management.

**Q: Does this work with CTest/CMake?**

A: Yes! The tests are still Google Test cases, so they work with standard test runners.

**Q: Should I use the macros or the manual API?**

A: Use the macros (`RUN_ISOLATED_TEST`, `RUN_ISOLATED_TESTS`, etc.) for most cases - they're simpler and less error-prone. Use the manual API (`registerTest()` + `executeAllTests()`) only when you need more control over the registration/execution flow, such as:
- Dynamically generating test configurations at runtime
- Sharing test registration logic across multiple TEST blocks
- Advanced control flow scenarios

**Q: Do tests run automatically after registration, or do I need to call executeAllTests()?**

A: **You MUST call `executeAllTests()` explicitly.** Tests do NOT run automatically. If you forget to call it, your tests will be silently ignored. Always follow this pattern:

```cpp
TEST(MyTest, IsolatedTests) {
  ProcessIsolatedTestRunner::registerTest("MyTest", []() { /* ... */ });

  // ✅ REQUIRED: Execute the tests
  bool passed = ProcessIsolatedTestRunner::executeAllTests();
  EXPECT_TRUE(passed);
}
```

**Q: How can I detect if I forgot to execute registered tests?**

A: After `executeAllTests()`, verify that `getTestResults().size()` matches your expected test count:

```cpp
// Register N tests
ProcessIsolatedTestRunner::registerTest("Test1", []() { /* ... */ });
ProcessIsolatedTestRunner::registerTest("Test2", []() { /* ... */ });

// Execute
bool passed = ProcessIsolatedTestRunner::executeAllTests();

// Verify count
auto results = ProcessIsolatedTestRunner::getTestResults();
EXPECT_EQ(results.size(), 2) << "Expected 2 tests to run";
```

**Q: Do I need to call clear() manually?**

A: No. The `clear()` method is only useful for advanced use cases where you need to clear tests that were registered but never executed. If you manually call `clear()` when tests were registered but not executed, it will warn you:

```
⚠️  WARNING: ProcessIsolatedTestRunner::clear() called with 2 unexecuted test(s)!
   Registered: 2 test(s)
   Executed:   0 test(s)
   Did you forget to call executeAllTests()?
```

---

## See Also

- **ProcessIsolatedTestRunner.hpp** - Full API documentation
- **ProcessIsolatedTestRunner.cpp** - Implementation details
- **RcclWrapTests.cpp** - Usage examples
