# C++ Testability

Every C++ component must be unit testable without a GPU, without the
rocprofiler-sdk runtime, and without touching the real filesystem.

## Inject dependencies through a virtual interface

A class does not build its own dependencies. It takes them.

```cpp
// Bad: creates the thing it depends on, so a test cannot replace it
class OrderProcessor {
    Database m_db;
public:
    void process(const Order& o) { m_db.save(o); }
};

// Good: the dependency arrives through an interface
class Database {
public:
    virtual ~Database() = default;
    virtual void save(const Order& o) = 0;
};

class OrderProcessor {
    Database& m_db;
public:
    explicit OrderProcessor(Database& db) : m_db(db) {}
    void process(const Order& o) { m_db.save(o); }
};
```

This is the pattern the tree already uses. `SdkWrapper` and `SdkWrapperImpl` in
`src/lib/rocprofiler_compute_tool/sdk_wrapper.h` hide the rocprofiler-sdk calls
behind an interface. `InputParameters` and `EnvInputParameters` hide the
environment. `CountersWriter` and `CsvCountersWriter` hide the output format.
`filesystem_wrapper_t` and `filesystem_wrapper_impl_t` in
`src/lib/pc_sampling_collector/filesystem_wrapper.h` hide disk access.

Follow it. Do not replace a virtual interface with a template policy parameter:
the existing mocks derive from these interfaces and would stop working.

## What must be behind a seam

Anything the test cannot control on its own:

- The rocprofiler-sdk and any other host API.
- The filesystem, the environment block, the network.
- The clock, and anything random.

Everything else should be a pure function where it can be. Same input, same
output, nothing to mock.

## Process-lifetime state

Some state has to outlive every object we own, because the host callback API
gives us nowhere to put it. That is allowed under the rules in
[`callback-boundaries.md`](callback-boundaries.md). The requirement this file
adds is narrow: the accessor must stay injectable, so a test can substitute its
own instance. `EnvInputParameters` takes its `EnvironCache` as a constructor
parameter defaulted to `EnvironCache::instance()`. Production code gets the
shared instance without saying so, and a test passes its own. Do not call an
accessor like `instance()` from the middle of a function, where nothing can
replace it.

## gtest and gmock conventions

One test target per library, with its own `main.cpp`.

Fixtures live in `test_<unit>.h` and derive from `::testing::Test`. The test
bodies live in the matching `test_<unit>.cpp`.

Mocks live in `mocks.h` and `mocks.cpp` next to the tests, and derive from the
production interface.

Name a test `TEST_F(TestSubject, Condition_ExpectedBehaviour)`. The condition
comes first, the expected result second, as in
`TEST_F(TestCountersWriter, SinkFailureMidStream_StopsAndIsReported)`. The name
already describes the case, so the test body needs no comment saying the same
thing.

## Checklist

- [ ] No hidden dependency on the sdk, filesystem, environment, clock, or
      network
- [ ] Dependencies injected through a virtual interface, taken by reference
- [ ] Pure functions where the work allows it
- [ ] Single responsibility, which is what makes a class testable at all
- [ ] Test names say condition and expectation
