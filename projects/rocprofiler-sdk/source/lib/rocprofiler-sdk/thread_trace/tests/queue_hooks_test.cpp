// projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/thread_trace/tests/queue_hooks_test.cpp
#include "lib/rocprofiler-sdk/thread_trace/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp"

#include <gtest/gtest.h>

// Positive-path test (DispatchThreadTracer activation -> is_any_active() == true,
// write_hook() pushing inst_pkt entries, signal_completion_hook() draining them)
// is DEFERRED to C2 integration smoke (rocprofv3 --att). Synthesizing a
// DispatchThreadTracer activation in a unit-test fixture requires full HSA queue
// plumbing (queue_controller, agent_cache, AQL packet construction) that is not
// available without an integration test environment.

namespace
{
TEST(ThreadTraceQueueHooks, IsAnyActiveReturnsFalseWhenNoContextActive)
{
    // No thread_trace context is active in this fresh test.
    EXPECT_FALSE(rocprofiler::thread_trace::is_any_active());
}
}  // namespace
