// projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/counters/tests/queue_hooks_test.cpp
#include "lib/rocprofiler-sdk/counters/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp"

#include <gtest/gtest.h>

namespace
{
TEST(CountersQueueHooks, IsAnyActiveReturnsFalseWhenNoContextActive)
{
    // No counter collection context is active in this fresh test.
    EXPECT_FALSE(rocprofiler::counters::is_any_active());
}
}  // namespace
