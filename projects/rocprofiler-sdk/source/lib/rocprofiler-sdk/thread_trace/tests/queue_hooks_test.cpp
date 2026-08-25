#include "lib/rocprofiler-sdk/thread_trace/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp"

#include <gtest/gtest.h>

// Positive-path coverage (active tracer + write/completion hooks) is provided
// by the rocprofv3 --att integration smoke; standalone DispatchThreadTracer
// activation requires full HSA queue plumbing unavailable in unit tests.

namespace
{
TEST(ThreadTraceQueueHooks, IsAnyActiveReturnsFalseWhenNoContextActive)
{
    EXPECT_FALSE(rocprofiler::thread_trace::is_any_active());
}
}  // namespace
