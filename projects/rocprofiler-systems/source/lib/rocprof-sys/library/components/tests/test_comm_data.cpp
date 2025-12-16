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

#include "core/state.hpp"
#include "rocprof-sys/library/components/comm_data.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <string>

namespace rocprofsys
{
namespace component
{
namespace testing
{

class comm_data_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize test data
        test_gotcha_data.tool_id = "test_comm_function";
        test_ep                  = reinterpret_cast<void*>(0x12345678);
        test_buffer              = reinterpret_cast<void*>(0x87654321);
        test_const_buffer        = reinterpret_cast<const void*>(0x11111111);
        test_count               = 1024;
        test_tag                 = 0xABCDEF;
        test_tag_mask            = 0xFFFFFF;
        test_remote_addr         = 0x1122334455667788ULL;
        test_rkey                = reinterpret_cast<void*>(0xAAA);
        test_param               = reinterpret_cast<const void*>(0x999);
        test_length              = 512;
        test_length_ptr          = &test_length;
        test_id                  = 42;
        test_header              = reinterpret_cast<const void*>(0xBBB);
        test_header_length       = 16;
    }

    void TearDown() override
    {
        // Cleanup if needed
    }

    // Test data
    tim::component::gotcha_data test_gotcha_data;
    void*                       test_ep;
    void*                       test_buffer;
    const void*                 test_const_buffer;
    size_t                      test_count;
    uint64_t                    test_tag;
    uint64_t                    test_tag_mask;
    uint64_t                    test_remote_addr;
    void*                       test_rkey;
    const void*                 test_param;
    size_t                      test_length;
    size_t*                     test_length_ptr;
    unsigned                    test_id;
    const void*                 test_header;
    size_t                      test_header_length;
};

TEST_F(comm_data_test, static_labels)
{
    EXPECT_STREQ(comm_data::ucx_send::label, "UCX Comm Send");
    EXPECT_STREQ(comm_data::ucx_recv::label, "UCX Comm Recv");
    EXPECT_STREQ(comm_data::ucx_send::value, "comm_data");
    EXPECT_STREQ(comm_data::ucx_recv::value, "comm_data");
}

TEST_F(comm_data_test, component_lifecycle)
{
    EXPECT_NO_THROW(comm_data::preinit());
    EXPECT_NO_THROW(comm_data::configure());
    EXPECT_NO_THROW(comm_data::start());
    EXPECT_NO_THROW(comm_data::stop());
    EXPECT_NO_THROW(comm_data::global_finalize());
}

TEST_F(comm_data_test, ucx_tag_send_nbx_audit)
{
    // Test ucp_tag_send_nbx auditing
    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_const_buffer, test_count, test_tag,
                                     test_param));
}

TEST_F(comm_data_test, ucx_tag_recv_nbx_audit)
{
    // Test ucp_tag_recv_nbx auditing
    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_buffer, test_count, test_tag, test_tag_mask,
                                     test_param));
}

TEST_F(comm_data_test, ucx_put_nbx_audit)
{
    // Test ucp_put_nbx auditing
    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_const_buffer, test_count, test_remote_addr,
                                     test_rkey, test_param));
}

TEST_F(comm_data_test, ucx_get_nbx_audit)
{
    // Test ucp_get_nbx auditing
    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_buffer, test_count, test_remote_addr, test_rkey,
                                     test_param));
}

TEST_F(comm_data_test, ucx_am_send_nbx_audit)
{
    // Test ucp_am_send_nbx auditing
    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_id, test_header, test_header_length,
                                     test_const_buffer, test_count, test_param));
}

TEST_F(comm_data_test, ucx_stream_send_nbx_audit)
{
    // Test ucp_stream_send_nbx auditing
    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_const_buffer, test_count, test_param));
}

TEST_F(comm_data_test, ucx_stream_recv_nbx_audit)
{
    // Test ucp_stream_recv_nbx auditing
    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_buffer, test_count, test_length_ptr,
                                     test_param));
}

TEST_F(comm_data_test, ucx_legacy_tag_send_audit)
{
    // Test legacy ucp_tag_send_nb auditing
    void* tag_ptr = reinterpret_cast<void*>(test_tag);
    void* cb_ptr  = reinterpret_cast<void*>(0xCCC);

    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_count, test_buffer, tag_ptr, cb_ptr));
}

TEST_F(comm_data_test, ucx_legacy_tag_recv_audit)
{
    // Test legacy ucp_tag_recv_nb auditing
    void* tag_ptr      = reinterpret_cast<void*>(test_tag);
    void* tag_mask_ptr = reinterpret_cast<void*>(test_tag_mask);
    void* req_ptr      = reinterpret_cast<void*>(0xDDD);
    void* cb_ptr       = reinterpret_cast<void*>(0xCCC);

    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_count, test_buffer, tag_ptr, tag_mask_ptr,
                                     req_ptr, cb_ptr));
}

TEST_F(comm_data_test, ucx_legacy_rma_audit)
{
    // Test legacy RMA operations auditing
    void* cb_ptr = reinterpret_cast<void*>(0xCCC);

    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_count, test_remote_addr, test_rkey, cb_ptr));
}

TEST_F(comm_data_test, ucx_legacy_am_send_audit)
{
    // Test legacy AM send auditing
    void*    header_ptr = reinterpret_cast<void*>(0xEEE);
    void*    data_ptr   = reinterpret_cast<void*>(0xFFF);
    void*    cb_ptr     = reinterpret_cast<void*>(0xCCC);
    unsigned flags      = 0;

    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_id, header_ptr, test_header_length, data_ptr,
                                     test_count, flags, cb_ptr));
}

TEST_F(comm_data_test, ucx_legacy_stream_audit)
{
    // Test legacy stream operations auditing
    void*    req_ptr = reinterpret_cast<void*>(0xDDD);
    void*    cb_ptr  = reinterpret_cast<void*>(0xCCC);
    unsigned flags   = 0;

    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_buffer, test_count, req_ptr, flags, cb_ptr));
}

TEST_F(comm_data_test, zero_count_operations)
{
    // Test operations with zero count
    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_const_buffer, size_t(0), test_tag, test_param));
}

TEST_F(comm_data_test, large_count_operations)
{
    // Test operations with large count
    size_t large_count = 1024ULL * 1024ULL * 1024ULL;  // 1GB
    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_const_buffer, large_count, test_tag,
                                     test_param));
}

TEST_F(comm_data_test, null_pointer_operations)
{
    // Test operations with null pointers (should not crash)
    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, nullptr,
                                     test_const_buffer, test_count, test_tag,
                                     test_param));

    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     static_cast<const void*>(nullptr), test_count,
                                     test_tag, test_param));
}

TEST_F(comm_data_test, extreme_tag_values)
{
    // Test with extreme tag values
    uint64_t max_tag  = UINT64_MAX;
    uint64_t zero_tag = 0;

    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_const_buffer, test_count, max_tag, test_param));

    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_const_buffer, test_count, zero_tag,
                                     test_param));
}

TEST_F(comm_data_test, extreme_remote_addresses)
{
    // Test with extreme remote addresses
    uint64_t max_addr  = UINT64_MAX;
    uint64_t zero_addr = 0;

    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_const_buffer, test_count, max_addr, test_rkey,
                                     test_param));

    EXPECT_NO_THROW(comm_data::audit(test_gotcha_data, audit::incoming{}, test_ep,
                                     test_const_buffer, test_count, zero_addr, test_rkey,
                                     test_param));
}

TEST_F(comm_data_test, different_gotcha_tool_ids)
{
    // Test with different tool IDs
    tim::component::gotcha_data tag_send_data;
    tag_send_data.tool_id = "ucp_tag_send_nbx";

    tim::component::gotcha_data tag_recv_data;
    tag_recv_data.tool_id = "ucp_tag_recv_nbx";

    tim::component::gotcha_data put_data;
    put_data.tool_id = "ucp_put_nbx";

    EXPECT_NO_THROW(comm_data::audit(tag_send_data, audit::incoming{}, test_ep,
                                     test_const_buffer, test_count, test_tag,
                                     test_param));

    EXPECT_NO_THROW(comm_data::audit(tag_recv_data, audit::incoming{}, test_ep,
                                     test_buffer, test_count, test_tag, test_tag_mask,
                                     test_param));

    EXPECT_NO_THROW(comm_data::audit(put_data, audit::incoming{}, test_ep,
                                     test_const_buffer, test_count, test_remote_addr,
                                     test_rkey, test_param));
}

}  // namespace testing
}  // namespace component
}  // namespace rocprofsys
