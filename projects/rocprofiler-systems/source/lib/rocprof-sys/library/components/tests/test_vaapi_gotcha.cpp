// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "gtest/gtest.h"

#include "common/tests/mock_trace_sink.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// Mock VAAPI types for testing
using VADisplay       = void*;
using VAContextID     = uint32_t;
using VASurfaceID     = uint32_t;
using VABufferID      = uint32_t;
using VAConfigID      = uint32_t;
using VAImageID       = uint32_t;
using VAProfile       = int;
using VAEntrypoint    = int;
using VABufferType    = int;
using VAStatus        = int;
using VASurfaceAttrib = int;
using VAConfigAttrib  = int;

namespace rocprofsys
{
namespace component
{
namespace audit
{
struct incoming
{};
struct outgoing
{};
}  // namespace audit

/**
 * @brief Mock vaapi_gotcha component for unit testing
 *
 * This mock implementation allows us to test the trace event emission logic
 * without requiring actual VAAPI runtime or gotcha library initialization.
 */
class vaapi_gotcha_mock
{
public:
    using trace_sink_ptr = std::shared_ptr<testing::mock_trace_sink>;

    vaapi_gotcha_mock()
    : m_sink(std::make_shared<testing::mock_trace_sink>())
    , m_is_started(false)
    {}

    explicit vaapi_gotcha_mock(trace_sink_ptr sink)
    : m_sink(std::move(sink))
    , m_is_started(false)
    {}

    // Lifecycle methods
    void start()
    {
        if(!m_is_started)
        {
            m_is_started = true;
        }
    }

    void stop() { m_is_started = false; }

    bool is_running() const { return m_is_started; }

    // Get the sink for verification
    trace_sink_ptr get_sink() const { return m_sink; }

    // Mock gotcha_data structure
    struct mock_gotcha_data
    {
        const char* tool_id;
        void*       wrapper_pointer;
        void*       wrappee_pointer;
    };

    // Audit functions that emit trace events for VAAPI calls
    void audit_vaBeginPicture(const mock_gotcha_data& data, audit::incoming,
                              VADisplay dpy, VAContextID context,
                              VASurfaceID render_target)
    {
        if(!m_is_started || !m_sink) return;

        m_sink->emit_begin_with_args(data.tool_id, "vaapi", "dpy", ptr_to_str(dpy),
                                     "context", context, "render_target", render_target);
    }

    void audit_vaBeginPicture(const mock_gotcha_data& data, audit::outgoing)
    {
        if(!m_is_started || !m_sink) return;
        m_sink->emit_end(data.tool_id, "vaapi");
    }

    void audit_vaCreateBuffer(const mock_gotcha_data& data, audit::incoming,
                              VADisplay dpy, VAContextID context, VABufferType type,
                              unsigned int size, unsigned int num_elements,
                              void* buffer_data, VABufferID* buf_id)
    {
        if(!m_is_started || !m_sink) return;

        m_sink->emit_begin_with_args(data.tool_id, "vaapi", "dpy", ptr_to_str(dpy),
                                     "context", context, "buffer_type", type, "size",
                                     size, "num_elements", num_elements);
    }

    void audit_vaCreateBuffer(const mock_gotcha_data& data, audit::outgoing)
    {
        if(!m_is_started || !m_sink) return;
        m_sink->emit_end(data.tool_id, "vaapi");
    }

    void audit_vaCreateSurfaces(const mock_gotcha_data& data, audit::incoming,
                                VADisplay dpy, unsigned int format, unsigned int width,
                                unsigned int height, VASurfaceID* surfaces,
                                unsigned int num_surfaces, VASurfaceAttrib* attrib_list,
                                unsigned int num_attribs)
    {
        if(!m_is_started || !m_sink) return;

        m_sink->emit_begin_with_args(data.tool_id, "vaapi", "dpy", ptr_to_str(dpy),
                                     "format", format, "width", width, "height", height,
                                     "num_surfaces", num_surfaces);
    }

    void audit_vaCreateSurfaces(const mock_gotcha_data& data, audit::outgoing)
    {
        if(!m_is_started || !m_sink) return;
        m_sink->emit_end(data.tool_id, "vaapi");
    }

    void audit_vaSyncSurface(const mock_gotcha_data& data, audit::incoming, VADisplay dpy,
                             VASurfaceID surface)
    {
        if(!m_is_started || !m_sink) return;

        m_sink->emit_begin_with_args(data.tool_id, "vaapi", "dpy", ptr_to_str(dpy),
                                     "surface", surface);
    }

    void audit_vaSyncSurface(const mock_gotcha_data& data, audit::outgoing)
    {
        if(!m_is_started || !m_sink) return;
        m_sink->emit_end(data.tool_id, "vaapi");
    }

private:
    static std::string ptr_to_str(void* ptr)
    {
        char buf[32];
        snprintf(buf, sizeof(buf), "%p", ptr);
        return std::string(buf);
    }

    trace_sink_ptr m_sink;
    bool           m_is_started;
};

}  // namespace component
}  // namespace rocprofsys

//======================================================================================//
//
//                                  TEST FIXTURE
//
//======================================================================================//

namespace rocprofsys
{
namespace testing
{
/**
 * @brief Test fixture for VAAPI gotcha component tests
 *
 * Provides:
 * - Shared mock_trace_sink for event verification
 * - Shared vaapi_gotcha_mock instance
 * - Common setup/teardown logic
 * - Helper methods for test verification
 */
class VaapiGotchaTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create shared sink
        sink = std::make_shared<mock_trace_sink>();

        // Create gotcha mock with injected sink
        gotcha = std::make_unique<component::vaapi_gotcha_mock>(sink);

        // Set deterministic clock for testing
        test_timestamp = 1000;
        sink->set_clock([this]() { return test_timestamp++; });
    }

    void TearDown() override
    {
        gotcha.reset();
        sink.reset();
    }

    // Helper: Create mock gotcha data
    component::vaapi_gotcha_mock::mock_gotcha_data create_gotcha_data(const char* name)
    {
        return component::vaapi_gotcha_mock::mock_gotcha_data{ name, nullptr, nullptr };
    }

    // Helper: Verify begin/end pair
    void verify_begin_end_pair(const std::string& function_name)
    {
        ASSERT_GE(sink->get_event_count(), 2) << "Expected at least 2 events";

        auto events = sink->get_events_by_name(function_name);
        ASSERT_GE(events.size(), 2) << "Expected at least 2 events for " << function_name;

        EXPECT_EQ(events[0].phase, trace_phase::begin) << "First event should be begin";
        EXPECT_EQ(events[1].phase, trace_phase::end) << "Second event should be end";
        EXPECT_EQ(events[0].category, "vaapi");
        EXPECT_EQ(events[1].category, "vaapi");
    }

    // Helper: Verify event has expected arguments
    void verify_event_has_args(const trace_event&              evt,
                               const std::vector<std::string>& arg_keys)
    {
        for(const auto& key : arg_keys)
        {
            EXPECT_TRUE(evt.has_arg(key)) << "Event should have argument: " << key;
        }
    }

    // Shared test objects
    std::shared_ptr<mock_trace_sink>              sink;
    std::unique_ptr<component::vaapi_gotcha_mock> gotcha;
    uint64_t                                      test_timestamp;
};

//======================================================================================//
//
//                                  UNIT TESTS
//
//======================================================================================//

/**
 * @test NoEventBeforeStart
 * @brief Verify that no trace events are emitted before start() is called
 */
TEST_F(VaapiGotchaTest, NoEventBeforeStart)
{
    ASSERT_FALSE(gotcha->is_running());

    auto      data = create_gotcha_data("vaBeginPicture");
    VADisplay dpy  = reinterpret_cast<VADisplay>(0x1234);

    // Emit events without starting
    gotcha->audit_vaBeginPicture(data, component::audit::incoming{}, dpy, 1, 2);
    gotcha->audit_vaBeginPicture(data, component::audit::outgoing{});

    // No events should be captured
    EXPECT_EQ(sink->get_event_count(), 0);
    EXPECT_FALSE(sink->has_event("vaBeginPicture"));
}

/**
 * @test EmitTraceEventOnAPICall
 * @brief Verify that trace events are correctly emitted for VAAPI function calls
 */
TEST_F(VaapiGotchaTest, EmitTraceEventOnAPICall)
{
    gotcha->start();
    ASSERT_TRUE(gotcha->is_running());

    auto        data          = create_gotcha_data("vaBeginPicture");
    VADisplay   dpy           = reinterpret_cast<VADisplay>(0x1234);
    VAContextID context       = 42;
    VASurfaceID render_target = 99;

    // Simulate wrapped function call
    gotcha->audit_vaBeginPicture(data, component::audit::incoming{}, dpy, context,
                                 render_target);
    gotcha->audit_vaBeginPicture(data, component::audit::outgoing{});

    // Verify events were emitted
    EXPECT_EQ(sink->get_event_count(), 2);
    EXPECT_TRUE(sink->has_event("vaBeginPicture"));
    EXPECT_TRUE(sink->has_category("vaapi"));

    // Verify begin/end pairing
    verify_begin_end_pair("vaBeginPicture");

    // Verify arguments
    auto events = sink->get_events_by_name("vaBeginPicture");
    ASSERT_GE(events.size(), 1);
    verify_event_has_args(events[0], { "dpy", "context", "render_target" });

    // Verify argument values
    EXPECT_EQ(events[0].get_arg("context").value(), "42");
    EXPECT_EQ(events[0].get_arg("render_target").value(), "99");
}

/**
 * @test MultipleAPICallsEmitMultipleEvents
 * @brief Verify that multiple API calls emit the correct number of events
 */
TEST_F(VaapiGotchaTest, MultipleAPICallsEmitMultipleEvents)
{
    gotcha->start();

    VADisplay dpy = reinterpret_cast<VADisplay>(0x5678);

    // Call multiple VAAPI functions
    auto data1 = create_gotcha_data("vaBeginPicture");
    gotcha->audit_vaBeginPicture(data1, component::audit::incoming{}, dpy, 1, 10);
    gotcha->audit_vaBeginPicture(data1, component::audit::outgoing{});

    auto data2 = create_gotcha_data("vaCreateBuffer");
    gotcha->audit_vaCreateBuffer(data2, component::audit::incoming{}, dpy, 1, 3, 256, 4,
                                 nullptr, nullptr);
    gotcha->audit_vaCreateBuffer(data2, component::audit::outgoing{});

    auto data3 = create_gotcha_data("vaSyncSurface");
    gotcha->audit_vaSyncSurface(data3, component::audit::incoming{}, dpy, 100);
    gotcha->audit_vaSyncSurface(data3, component::audit::outgoing{});

    // Verify total event count (3 functions × 2 events each)
    EXPECT_EQ(sink->get_event_count(), 6);

    // Verify each function emitted events
    EXPECT_EQ(sink->count_events_by_name("vaBeginPicture"), 2);
    EXPECT_EQ(sink->count_events_by_name("vaCreateBuffer"), 2);
    EXPECT_EQ(sink->count_events_by_name("vaSyncSurface"), 2);

    // Verify event sequence
    EXPECT_TRUE(sink->verify_event_sequence({ "vaBeginPicture", "vaBeginPicture",
                                              "vaCreateBuffer", "vaCreateBuffer",
                                              "vaSyncSurface", "vaSyncSurface" }));

    // Verify all events are balanced
    EXPECT_TRUE(sink->verify_balanced_events());
}

/**
 * @test HandlesMultipleStartStopCycles
 * @brief Verify that start/stop cycles work correctly
 */
TEST_F(VaapiGotchaTest, HandlesMultipleStartStopCycles)
{
    auto      data = create_gotcha_data("vaCreateSurfaces");
    VADisplay dpy  = reinterpret_cast<VADisplay>(0xABCD);

    // First cycle: start -> emit -> stop
    gotcha->start();
    ASSERT_TRUE(gotcha->is_running());

    gotcha->audit_vaCreateSurfaces(data, component::audit::incoming{}, dpy, 1, 1920, 1080,
                                   nullptr, 1, nullptr, 0);
    gotcha->audit_vaCreateSurfaces(data, component::audit::outgoing{});

    EXPECT_EQ(sink->get_event_count(), 2);

    gotcha->stop();
    ASSERT_FALSE(gotcha->is_running());

    // Events should not be emitted when stopped
    gotcha->audit_vaCreateSurfaces(data, component::audit::incoming{}, dpy, 1, 1920, 1080,
                                   nullptr, 1, nullptr, 0);
    gotcha->audit_vaCreateSurfaces(data, component::audit::outgoing{});

    EXPECT_EQ(sink->get_event_count(), 2);  // Still 2, no new events

    // Second cycle: start again
    gotcha->start();
    ASSERT_TRUE(gotcha->is_running());

    gotcha->audit_vaCreateSurfaces(data, component::audit::incoming{}, dpy, 2, 3840, 2160,
                                   nullptr, 2, nullptr, 0);
    gotcha->audit_vaCreateSurfaces(data, component::audit::outgoing{});

    EXPECT_EQ(sink->get_event_count(), 4);  // Now 4 events total

    // Verify both calls have correct arguments
    auto events = sink->get_events_by_name("vaCreateSurfaces");
    ASSERT_EQ(events.size(), 4);

    // First call: 1920x1080
    EXPECT_EQ(events[0].get_arg("width").value(), "1920");
    EXPECT_EQ(events[0].get_arg("height").value(), "1080");

    // Second call: 3840x2160
    EXPECT_EQ(events[2].get_arg("width").value(), "3840");
    EXPECT_EQ(events[2].get_arg("height").value(), "2160");
}

/**
 * @test EventTimestampsAreMonotonic
 * @brief Verify that event timestamps are monotonically increasing
 */
TEST_F(VaapiGotchaTest, EventTimestampsAreMonotonic)
{
    gotcha->start();

    auto      data = create_gotcha_data("vaBeginPicture");
    VADisplay dpy  = reinterpret_cast<VADisplay>(0x1111);

    // Emit multiple events
    for(int i = 0; i < 5; ++i)
    {
        gotcha->audit_vaBeginPicture(data, component::audit::incoming{}, dpy, i, i * 10);
        gotcha->audit_vaBeginPicture(data, component::audit::outgoing{});
    }

    // Verify timestamps are monotonic
    const auto& events = sink->get_events();
    ASSERT_EQ(events.size(), 10);

    for(size_t i = 1; i < events.size(); ++i)
    {
        EXPECT_LT(events[i - 1].timestamp, events[i].timestamp)
            << "Timestamps should be monotonically increasing at index " << i;
    }
}

/**
 * @test NullSinkHandledGracefully
 * @brief Verify that null sink is handled without crashes
 */
TEST_F(VaapiGotchaTest, NullSinkHandledGracefully)
{
    // Create gotcha with null sink
    auto gotcha_null = std::make_unique<component::vaapi_gotcha_mock>(nullptr);

    gotcha_null->start();

    auto      data = create_gotcha_data("vaBeginPicture");
    VADisplay dpy  = reinterpret_cast<VADisplay>(0x1234);

    // Should not crash
    EXPECT_NO_THROW({
        gotcha_null->audit_vaBeginPicture(data, component::audit::incoming{}, dpy, 1, 2);
        gotcha_null->audit_vaBeginPicture(data, component::audit::outgoing{});
    });
}

/**
 * @test RepeatedAPICallsWithSameParameters
 * @brief Verify that repeated calls with identical parameters work correctly
 */
TEST_F(VaapiGotchaTest, RepeatedAPICallsWithSameParameters)
{
    gotcha->start();

    auto        data    = create_gotcha_data("vaSyncSurface");
    VADisplay   dpy     = reinterpret_cast<VADisplay>(0xCAFE);
    VASurfaceID surface = 777;

    // Call same function multiple times with identical parameters
    constexpr int num_calls = 3;
    for(int i = 0; i < num_calls; ++i)
    {
        gotcha->audit_vaSyncSurface(data, component::audit::incoming{}, dpy, surface);
        gotcha->audit_vaSyncSurface(data, component::audit::outgoing{});
    }

    // Verify correct number of events
    EXPECT_EQ(sink->get_event_count(), num_calls * 2);
    EXPECT_EQ(sink->count_events_by_name("vaSyncSurface"), num_calls * 2);

    // Verify all events have same parameters
    auto events = sink->get_events_by_name("vaSyncSurface");
    for(const auto& evt : events)
    {
        if(evt.phase == trace_phase::begin)
        {
            EXPECT_EQ(evt.get_arg("surface").value(), "777");
        }
    }
}

/**
 * @test EventCategoryIsCorrect
 * @brief Verify that all events have the correct category
 */
TEST_F(VaapiGotchaTest, EventCategoryIsCorrect)
{
    gotcha->start();

    VADisplay dpy = reinterpret_cast<VADisplay>(0x2222);

    // Emit various VAAPI calls
    auto data1 = create_gotcha_data("vaBeginPicture");
    gotcha->audit_vaBeginPicture(data1, component::audit::incoming{}, dpy, 1, 2);
    gotcha->audit_vaBeginPicture(data1, component::audit::outgoing{});

    auto data2 = create_gotcha_data("vaCreateBuffer");
    gotcha->audit_vaCreateBuffer(data2, component::audit::incoming{}, dpy, 1, 0, 100, 1,
                                 nullptr, nullptr);
    gotcha->audit_vaCreateBuffer(data2, component::audit::outgoing{});

    auto data3 = create_gotcha_data("vaCreateSurfaces");
    gotcha->audit_vaCreateSurfaces(data3, component::audit::incoming{}, dpy, 1, 640, 480,
                                   nullptr, 1, nullptr, 0);
    gotcha->audit_vaCreateSurfaces(data3, component::audit::outgoing{});

    // All events should have "vaapi" category
    const auto& events = sink->get_events();
    ASSERT_EQ(events.size(), 6);

    for(const auto& evt : events)
    {
        EXPECT_EQ(evt.category, "vaapi") << "Event " << evt.name << " has wrong category";
    }

    EXPECT_TRUE(sink->has_category("vaapi"));
}

/**
 * @test BeginEndEventsAreProperlyCounted
 * @brief Verify that begin and end events are properly counted
 */
TEST_F(VaapiGotchaTest, BeginEndEventsAreProperlyCounted)
{
    gotcha->start();

    auto      data = create_gotcha_data("vaBeginPicture");
    VADisplay dpy  = reinterpret_cast<VADisplay>(0x3333);

    constexpr int num_pairs = 4;
    for(int i = 0; i < num_pairs; ++i)
    {
        gotcha->audit_vaBeginPicture(data, component::audit::incoming{}, dpy, i, i * 100);
        gotcha->audit_vaBeginPicture(data, component::audit::outgoing{});
    }

    // Count begin and end events
    size_t begin_count = sink->count_events_by_phase(trace_phase::begin);
    size_t end_count   = sink->count_events_by_phase(trace_phase::end);

    EXPECT_EQ(begin_count, num_pairs);
    EXPECT_EQ(end_count, num_pairs);
    EXPECT_EQ(begin_count, end_count);
}
}  // namespace testing
}  // namespace rocprofsys

//======================================================================================//
//
//                                  MAIN
//
//======================================================================================//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}