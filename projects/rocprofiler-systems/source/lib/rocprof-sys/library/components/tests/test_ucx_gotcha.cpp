// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "core/config.hpp"
#include "rocprof-sys/library/components/ucx_gotcha.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <string>

namespace rocprofsys
{
namespace component
{
namespace testing
{

class ucx_gotcha_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize test data
        test_gotcha_data.tool_id = "test_ucx_function";
        test_ep                  = reinterpret_cast<void*>(0x12345678);
        test_buffer              = reinterpret_cast<void*>(0x87654321);
        test_count               = 1024;
        test_tag                 = 0xABCDEF;
        test_remote_addr         = 0x1122334455667788ULL;
    }

    void TearDown() override
    {
        // Stop any running gotcha
        ucx_gotcha::stop();
    }

    // Test data
    tim::component::gotcha_data test_gotcha_data;
    void*                       test_ep;
    void*                       test_buffer;
    size_t                      test_count;
    uint64_t                    test_tag;
    uint64_t                    test_remote_addr;
};

TEST_F(ucx_gotcha_test, component_label) { EXPECT_EQ(ucx_gotcha::label(), "ucx_gotcha"); }

TEST_F(ucx_gotcha_test, configure_function_called)
{
    // Configure should not throw
    EXPECT_NO_THROW(ucx_gotcha::configure());
}

TEST_F(ucx_gotcha_test, start_function_called) { EXPECT_NO_THROW(ucx_gotcha::start()); }

TEST_F(ucx_gotcha_test, stop_function_called) { EXPECT_NO_THROW(ucx_gotcha::stop()); }

TEST_F(ucx_gotcha_test, shutdown_function_called)
{
    EXPECT_NO_THROW(ucx_gotcha::shutdown());
}

TEST_F(ucx_gotcha_test, audit_incoming_no_args)
{
    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}));
}

TEST_F(ucx_gotcha_test, audit_incoming_single_arg)
{
    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep));
}

TEST_F(ucx_gotcha_test, audit_incoming_multiple_args)
{
    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                      test_buffer, reinterpret_cast<void*>(test_count)));
}

TEST_F(ucx_gotcha_test, audit_tag_send_nbx)
{
    const void* param = reinterpret_cast<const void*>(0x999);

    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                      test_buffer, test_count, test_tag, param));
}

TEST_F(ucx_gotcha_test, audit_tag_recv_nbx)
{
    uint64_t    tag_mask = 0xFFFFFF;
    const void* param    = reinterpret_cast<const void*>(0x999);

    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                      test_buffer, test_count, test_tag, tag_mask,
                                      param));
}

TEST_F(ucx_gotcha_test, audit_put_nbx)
{
    void*       rkey  = reinterpret_cast<void*>(0xAAA);
    const void* param = reinterpret_cast<const void*>(0x999);

    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                      test_buffer, test_count, test_remote_addr, rkey,
                                      param));
}

TEST_F(ucx_gotcha_test, audit_get_nbx)
{
    void*       rkey  = reinterpret_cast<void*>(0xAAA);
    const void* param = reinterpret_cast<const void*>(0x999);

    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                      test_buffer, test_count, test_remote_addr, rkey,
                                      param));
}

TEST_F(ucx_gotcha_test, audit_am_send_nbx)
{
    unsigned    id            = 42;
    const void* header        = reinterpret_cast<const void*>(0xBBB);
    size_t      header_length = 16;
    const void* param         = reinterpret_cast<const void*>(0x999);

    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep, id,
                                      header, header_length, test_buffer, test_count,
                                      param));
}

TEST_F(ucx_gotcha_test, audit_stream_send_nbx)
{
    const void* param = reinterpret_cast<const void*>(0x999);

    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                      test_buffer, test_count, param));
}

TEST_F(ucx_gotcha_test, audit_stream_recv_nbx)
{
    size_t      length     = 512;
    size_t*     length_ptr = &length;
    const void* param      = reinterpret_cast<const void*>(0x999);

    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                      test_buffer, test_count, length_ptr, param));
}

TEST_F(ucx_gotcha_test, audit_outgoing_void_ptr)
{
    void* return_val = reinterpret_cast<void*>(0xDEADBEEF);

    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::outgoing{}, return_val));
}

TEST_F(ucx_gotcha_test, audit_outgoing_int)
{
    int return_val = 42;

    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::outgoing{}, return_val));
}

// Edge case tests
TEST_F(ucx_gotcha_test, zero_size_message_handling)
{
    // Test zero-size messages should return early without processing
    const void* header        = nullptr;
    size_t      header_length = 0;
    size_t      zero_count    = 0;
    unsigned    id            = 42;
    const void* param         = reinterpret_cast<const void*>(0x999);

    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep, id,
                                      header, header_length, test_buffer, zero_count,
                                      param));
}

TEST_F(ucx_gotcha_test, audit_null_buffer_zero_size)
{
    // Test with null buffer (valid for zero-size messages)
    size_t      zero_count = 0;
    const void* param      = reinterpret_cast<const void*>(0x999);

    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                      nullptr, zero_count, test_tag, param));
}

TEST_F(ucx_gotcha_test, audit_large_message_size)
{
    // Test with large message size (1 GB)
    size_t      large_count = 1024 * 1024 * 1024;
    void*       rkey        = reinterpret_cast<void*>(0xAAA);
    const void* param       = reinterpret_cast<const void*>(0x999);

    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                      test_buffer, large_count, test_remote_addr, rkey,
                                      param));
}

TEST_F(ucx_gotcha_test, audit_maximum_tag_value)
{
    // Test with maximum tag value
    uint64_t    max_tag = UINT64_MAX;
    const void* param   = reinterpret_cast<const void*>(0x999);

    EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                      test_buffer, test_count, max_tag, param));
}

TEST_F(ucx_gotcha_test, multiple_consecutive_audits)
{
    // Test multiple consecutive audit calls
    const void* param = reinterpret_cast<const void*>(0x999);

    for(int i = 0; i < 100; ++i)
    {
        EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                          test_buffer, test_count, test_tag, param));
    }
}

TEST_F(ucx_gotcha_test, interleaved_send_recv_audits)
{
    // Test interleaved send and recv audits
    uint64_t    tag_mask = 0xFFFFFFFFUL;
    const void* param    = reinterpret_cast<const void*>(0x999);

    for(int i = 0; i < 10; ++i)
    {
        // Send audit (5 args)
        EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                          test_buffer, test_count, test_tag, param));

        // Recv audit (6 args)
        EXPECT_NO_THROW(ucx_gotcha::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                          test_buffer, test_count, test_tag, tag_mask,
                                          param));
    }
}

// Idempotence tests
TEST_F(ucx_gotcha_test, start_idempotence)
{
    EXPECT_NO_THROW(ucx_gotcha::start());
    EXPECT_NO_THROW(ucx_gotcha::start());  // Second call should be safe
}

TEST_F(ucx_gotcha_test, shutdown_idempotence)
{
    EXPECT_NO_THROW(ucx_gotcha::shutdown());
    EXPECT_NO_THROW(ucx_gotcha::shutdown());
    EXPECT_NO_THROW(ucx_gotcha::shutdown());
}

TEST_F(ucx_gotcha_test, start_stop_idempotence)
{
    EXPECT_NO_THROW(ucx_gotcha::start());
    EXPECT_NO_THROW(ucx_gotcha::start());
    EXPECT_NO_THROW(ucx_gotcha::stop());
    EXPECT_NO_THROW(ucx_gotcha::stop());
}

}  // namespace testing
}  // namespace component
}  // namespace rocprofsys
