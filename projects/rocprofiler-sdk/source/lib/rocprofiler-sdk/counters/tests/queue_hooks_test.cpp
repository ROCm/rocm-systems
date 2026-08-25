#include "lib/rocprofiler-sdk/counters/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp"

#include <gtest/gtest.h>

namespace
{
TEST(CountersQueueHooks, IsAnyActiveReturnsFalseWhenNoContextActive)
{
    EXPECT_FALSE(rocprofiler::counters::is_any_active());
}
}  // namespace
