// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace
{

struct CaptureState
{
    std::mutex               mutex;
    std::vector<std::string> markers;
    std::size_t              pop_count = 0;
};

CaptureState& capture_state()
{
    // RecordFunction keeps collector callbacks until they are explicitly
    // removed, so the interception state must outlive static teardown.
    static auto* const state = new CaptureState{};
    return *state;
}

}  // namespace

#if defined(__GNUC__)
#    define TORCH_TRACE_TEST_EXPORT __attribute__((visibility("default")))
#else
#    define TORCH_TRACE_TEST_EXPORT
#endif

extern "C" TORCH_TRACE_TEST_EXPORT int roctxRangePushA(const char* message)
{
    auto&                             state = capture_state();
    const std::lock_guard<std::mutex> lock{state.mutex};
    state.markers.emplace_back(message == nullptr ? "" : message);
    return static_cast<int>(state.markers.size());
}

extern "C" TORCH_TRACE_TEST_EXPORT int roctxRangePop()
{
    auto&                             state = capture_state();
    const std::lock_guard<std::mutex> lock{state.mutex};
    ++state.pop_count;
    return 0;
}

extern "C" TORCH_TRACE_TEST_EXPORT void torch_trace_test_reset()
{
    auto&                             state = capture_state();
    const std::lock_guard<std::mutex> lock{state.mutex};
    state.markers.clear();
    state.pop_count = 0;
}

extern "C" TORCH_TRACE_TEST_EXPORT std::size_t torch_trace_test_push_count()
{
    auto&                             state = capture_state();
    const std::lock_guard<std::mutex> lock{state.mutex};
    return state.markers.size();
}

extern "C" TORCH_TRACE_TEST_EXPORT std::size_t torch_trace_test_pop_count()
{
    auto&                             state = capture_state();
    const std::lock_guard<std::mutex> lock{state.mutex};
    return state.pop_count;
}

extern "C" TORCH_TRACE_TEST_EXPORT std::size_t torch_trace_test_marker(std::size_t index,
                                                                       char*       output,
                                                                       std::size_t capacity)
{
    auto&                             state = capture_state();
    const std::lock_guard<std::mutex> lock{state.mutex};
    if (index >= state.markers.size())
    {
        return 0;
    }

    const std::string& marker   = state.markers[index];
    const std::size_t  required = marker.size() + 1;
    if (output == nullptr)
    {
        return required;
    }
    if (capacity < required)
    {
        return 0;
    }
    std::memcpy(output, marker.c_str(), required);
    return required;
}

#undef TORCH_TRACE_TEST_EXPORT
