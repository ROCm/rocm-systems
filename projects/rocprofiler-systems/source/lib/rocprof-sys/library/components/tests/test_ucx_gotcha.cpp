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

#include "rocprof-sys/library/components/ucx_gotcha.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

struct MockedItem
{
    int verbose = 0;
};

struct MockedGotchaData
{
    std::string tool_id;
};

struct GMockUCXGotcha
{
    MOCK_METHOD(void, configure, (std::string func_name));
    MOCK_METHOD(size_t, capacity, ());
    MOCK_METHOD(MockedItem*, at, (size_t index));
    MOCK_METHOD(void, disable, ());
    MOCK_METHOD(bool, get_is_running, (), (const));
    MOCK_METHOD(void, start, ());
    MOCK_METHOD(void, stop, ());
    MOCK_METHOD(std::function<void()>&, get_initializer, ());
};

struct GMockCommData
{
    // ucp_tag_send_nbx: (void* ep, const void* buffer, size_t count, uint64_t tag, const
    // void* param)
    MOCK_METHOD(void, audit_incoming_tag_send,
                (const MockedGotchaData& data, void* ep, const void* buffer, size_t count,
                 uint64_t tag, const void* param));

    // ucp_tag_recv_nbx: (void* worker, void* buffer, size_t count, uint64_t tag, uint64_t
    // tag_mask, const void* param)
    MOCK_METHOD(void, audit_incoming_tag_recv,
                (const MockedGotchaData& data, void* worker, void* buffer, size_t count,
                 uint64_t tag, uint64_t tag_mask, const void* param));

    // ucp_put_nbx: (void* ep, const void* buffer, size_t count, uint64_t remote_addr,
    // void* rkey, const void* param)
    MOCK_METHOD(void, audit_incoming_rma_put,
                (const MockedGotchaData& data, void* ep, const void* buffer, size_t count,
                 uint64_t remote_addr, void* rkey, const void* param));

    // ucp_get_nbx: (void* ep, void* buffer, size_t count, uint64_t remote_addr, void*
    // rkey, const void* param)
    MOCK_METHOD(void, audit_incoming_rma_get,
                (const MockedGotchaData& data, void* ep, void* buffer, size_t count,
                 uint64_t remote_addr, void* rkey, const void* param));

    // ucp_am_send_nbx: (void* ep, unsigned id, const void* header, size_t header_length,
    // const void* buffer, size_t count, const void* param)
    MOCK_METHOD(void, audit_incoming_am_send,
                (const MockedGotchaData& data, void* ep, unsigned id, const void* header,
                 size_t header_length, const void* buffer, size_t count,
                 const void* param));

    // ucp_stream_send_nbx: (void* ep, const void* buffer, size_t count, const void*
    // param)
    MOCK_METHOD(void, audit_incoming_stream_send,
                (const MockedGotchaData& data, void* ep, const void* buffer, size_t count,
                 const void* param));

    // ucp_stream_recv_nbx: (void* ep, void* buffer, size_t count, size_t* length, const
    // void* param)
    MOCK_METHOD(void, audit_incoming_stream_recv,
                (const MockedGotchaData& data, void* ep, void* buffer, size_t count,
                 size_t* length, const void* param));

    // outgoing audit with void* return
    MOCK_METHOD(void, audit_outgoing_ptr, (const MockedGotchaData& data, void* ret));

    // outgoing audit with int return
    MOCK_METHOD(void, audit_outgoing_int, (const MockedGotchaData& data, int ret));

    // get_initializer
    MOCK_METHOD(std::function<void()>&, get_initializer, ());
};

struct GMockCategoryRegion
{
    // Generic start
    MOCK_METHOD(void, start_generic, (std::string_view name));

    // ucp_tag_send_nbx: start(name, "ep", ep, "buffer", buffer, "count", count, "tag",
    // tag, "param", param)
    MOCK_METHOD(void, start_tag_send,
                (std::string_view name, void* ep, const void* buffer, size_t count,
                 uint64_t tag, const void* param));

    // ucp_tag_recv_nbx: start(name, "worker", worker, "buffer", buffer, "count", count,
    // "tag", tag, "tag_mask", tag_mask, "param", param)
    MOCK_METHOD(void, start_tag_recv,
                (std::string_view name, void* worker, void* buffer, size_t count,
                 uint64_t tag, uint64_t tag_mask, const void* param));

    // ucp_put_nbx: start(name, "ep", ep, "buffer", buffer, "count", count, "remote_addr",
    // remote_addr, "rkey", rkey, "param", param)
    MOCK_METHOD(void, start_rma_put,
                (std::string_view name, void* ep, const void* buffer, size_t count,
                 uint64_t remote_addr, void* rkey, const void* param));

    // ucp_get_nbx: start(name, "ep", ep, "buffer", buffer, "count", count, "remote_addr",
    // remote_addr, "rkey", rkey, "param", param)
    MOCK_METHOD(void, start_rma_get,
                (std::string_view name, void* ep, void* buffer, size_t count,
                 uint64_t remote_addr, void* rkey, const void* param));

    // ucp_am_send_nbx: start(name, "ep", ep, "id", id, "header", header, "header_length",
    // header_length, "buffer", buffer, "count", count, "param", param)
    MOCK_METHOD(void, start_am_send,
                (std::string_view name, void* ep, unsigned id, const void* header,
                 size_t header_length, const void* buffer, size_t count,
                 const void* param));

    // ucp_stream_send_nbx: start(name, "ep", ep, "buffer", buffer, "count", count,
    // "param", param)
    MOCK_METHOD(void, start_stream_send,
                (std::string_view name, void* ep, const void* buffer, size_t count,
                 const void* param));

    // ucp_stream_recv_nbx: start(name, "ep", ep, "buffer", buffer, "count", count,
    // "length", length, "param", param)
    MOCK_METHOD(void, start_stream_recv,
                (std::string_view name, void* ep, void* buffer, size_t count,
                 size_t* length, const void* param));

    // stop with void* return
    MOCK_METHOD(void, stop_ptr, (std::string_view name, void* ret));

    // stop with int return
    MOCK_METHOD(void, stop_int, (std::string_view name, int ret));
};

namespace test_globals
{
std::unique_ptr<GMockUCXGotcha>      g_ucx_gotcha_gmock;
std::unique_ptr<GMockCommData>       g_comm_data_gmock;
std::unique_ptr<GMockCategoryRegion> g_category_region_gmock;
}  // namespace test_globals

struct MockedUCXGotcha
{
    template <int N, typename... Args>
    static void configure(std::string func_name)
    {
        test_globals::g_ucx_gotcha_gmock->configure(func_name);
    }
    static size_t      capacity() { return test_globals::g_ucx_gotcha_gmock->capacity(); }
    static MockedItem* at(size_t index)
    {
        return test_globals::g_ucx_gotcha_gmock->at(index);
    }
    static void disable() { test_globals::g_ucx_gotcha_gmock->disable(); }
    bool        get_is_running() const
    {
        return test_globals::g_ucx_gotcha_gmock->get_is_running();
    }
    void                          start() { test_globals::g_ucx_gotcha_gmock->start(); }
    void                          stop() { test_globals::g_ucx_gotcha_gmock->stop(); }
    static std::function<void()>& get_initializer()
    {
        return test_globals::g_ucx_gotcha_gmock->get_initializer();
    }
};

template <typename GotchaData>
struct MockedCommData
{
    template <typename... Args>
    static void audit(const GotchaData&, tim::audit::incoming, Args&&...)
    {
        FAIL() << "Unexpected call of comm_data::audit(incoming)";
    }

    template <typename... Args>
    static void audit(const GotchaData&, tim::audit::outgoing, Args&&...)
    {
        FAIL() << "Unexpected call of comm_data::audit(outgoing)";
    }

    // ucp_tag_send_nbx
    static void audit(const GotchaData& _data, tim::audit::incoming, void* ep,
                      const void* buffer, size_t count, uint64_t tag, const void* param)
    {
        test_globals::g_comm_data_gmock->audit_incoming_tag_send(_data, ep, buffer, count,
                                                                 tag, param);
    }

    // ucp_tag_recv_nbx
    static void audit(const GotchaData& _data, tim::audit::incoming, void* worker,
                      void* buffer, size_t count, uint64_t tag, uint64_t tag_mask,
                      const void* param)
    {
        test_globals::g_comm_data_gmock->audit_incoming_tag_recv(
            _data, worker, buffer, count, tag, tag_mask, param);
    }

    // ucp_put_nbx
    static void audit(const GotchaData& _data, tim::audit::incoming, void* ep,
                      const void* buffer, size_t count, uint64_t remote_addr, void* rkey,
                      const void* param)
    {
        test_globals::g_comm_data_gmock->audit_incoming_rma_put(_data, ep, buffer, count,
                                                                remote_addr, rkey, param);
    }

    // ucp_get_nbx
    static void audit(const GotchaData& _data, tim::audit::incoming, void* ep,
                      void* buffer, size_t count, uint64_t remote_addr, void* rkey,
                      const void* param)
    {
        test_globals::g_comm_data_gmock->audit_incoming_rma_get(_data, ep, buffer, count,
                                                                remote_addr, rkey, param);
    }

    // ucp_am_send_nbx
    static void audit(const GotchaData& _data, tim::audit::incoming, void* ep,
                      unsigned id, const void* header, size_t header_length,
                      const void* buffer, size_t count, const void* param)
    {
        test_globals::g_comm_data_gmock->audit_incoming_am_send(
            _data, ep, id, header, header_length, buffer, count, param);
    }

    // ucp_stream_send_nbx
    static void audit(const GotchaData& _data, tim::audit::incoming, void* ep,
                      const void* buffer, size_t count, const void* param)
    {
        test_globals::g_comm_data_gmock->audit_incoming_stream_send(_data, ep, buffer,
                                                                    count, param);
    }

    // ucp_stream_recv_nbx
    static void audit(const GotchaData& _data, tim::audit::incoming, void* ep,
                      void* buffer, size_t count, size_t* length, const void* param)
    {
        test_globals::g_comm_data_gmock->audit_incoming_stream_recv(_data, ep, buffer,
                                                                    count, length, param);
    }

    // outgoing with void* return
    static void audit(const GotchaData& _data, tim::audit::outgoing, void* ret)
    {
        test_globals::g_comm_data_gmock->audit_outgoing_ptr(_data, ret);
    }

    // outgoing with int return
    static void audit(const GotchaData& _data, tim::audit::outgoing, int ret)
    {
        test_globals::g_comm_data_gmock->audit_outgoing_int(_data, ret);
    }
};

struct MockedCategoryRegion
{
    template <typename... Args>
    static void start(std::string_view, Args&&...)
    {
        FAIL() << "Unexpected call of category_region::start";
    }

    template <typename... Args>
    static void stop(std::string_view, Args&&...)
    {
        FAIL() << "Unexpected call of category_region::stop call";
    }

    // Generic start (no args)
    static void start(std::string_view name)
    {
        test_globals::g_category_region_gmock->start_generic(name);
    }

    // ucp_tag_send_nbx
    static void start(std::string_view name, const char*, void* ep, const char*,
                      const void* buffer, const char*, size_t count, const char*,
                      uint64_t tag, const char*, const void* param)
    {
        test_globals::g_category_region_gmock->start_tag_send(name, ep, buffer, count,
                                                              tag, param);
    }

    // ucp_tag_recv_nbx
    static void start(std::string_view name, const char*, void* worker, const char*,
                      void* buffer, const char*, size_t count, const char*, uint64_t tag,
                      const char*, uint64_t tag_mask, const char*, const void* param)
    {
        test_globals::g_category_region_gmock->start_tag_recv(name, worker, buffer, count,
                                                              tag, tag_mask, param);
    }

    // ucp_put_nbx
    static void start(std::string_view name, const char*, void* ep, const char*,
                      const void* buffer, const char*, size_t count, const char*,
                      uint64_t remote_addr, const char*, void* rkey, const char*,
                      const void* param)
    {
        test_globals::g_category_region_gmock->start_rma_put(name, ep, buffer, count,
                                                             remote_addr, rkey, param);
    }

    // ucp_get_nbx
    static void start(std::string_view name, const char*, void* ep, const char*,
                      void* buffer, const char*, size_t count, const char*,
                      uint64_t remote_addr, const char*, void* rkey, const char*,
                      const void* param)
    {
        test_globals::g_category_region_gmock->start_rma_get(name, ep, buffer, count,
                                                             remote_addr, rkey, param);
    }

    // ucp_am_send_nbx
    static void start(std::string_view name, const char*, void* ep, const char*,
                      unsigned id, const char*, const void* header, const char*,
                      size_t header_length, const char*, const void* buffer, const char*,
                      size_t count, const char*, const void* param)
    {
        test_globals::g_category_region_gmock->start_am_send(
            name, ep, id, header, header_length, buffer, count, param);
    }

    // ucp_stream_send_nbx
    static void start(std::string_view name, const char*, void* ep, const char*,
                      const void* buffer, const char*, size_t count, const char*,
                      const void* param)
    {
        test_globals::g_category_region_gmock->start_stream_send(name, ep, buffer, count,
                                                                 param);
    }

    // ucp_stream_recv_nbx
    static void start(std::string_view name, const char*, void* ep, const char*,
                      void* buffer, const char*, size_t count, const char*,
                      size_t* length, const char*, const void* param)
    {
        test_globals::g_category_region_gmock->start_stream_recv(name, ep, buffer, count,
                                                                 length, param);
    }

    // stop with void* return
    static void stop(std::string_view name, const char*, void* ret)
    {
        test_globals::g_category_region_gmock->stop_ptr(name, ret);
    }

    // stop with int return
    static void stop(std::string_view name, const char*, int ret)
    {
        test_globals::g_category_region_gmock->stop_int(name, ret);
    }
};

struct MockedUCXPolicy
{
    using gotcha_data     = MockedGotchaData;
    using comm_data       = MockedCommData<gotcha_data>;
    using category_region = MockedCategoryRegion;
    using ucx_bundle_t    = void;  // unused
    using ucx_gotcha_t    = MockedUCXGotcha;
};

using ucx_gotcha_under_test_t = rocprofsys::component::ucx_gotcha<MockedUCXPolicy>;

class ucx_gotcha_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        test_globals::g_ucx_gotcha_gmock      = std::make_unique<GMockUCXGotcha>();
        test_globals::g_comm_data_gmock       = std::make_unique<GMockCommData>();
        test_globals::g_category_region_gmock = std::make_unique<GMockCategoryRegion>();
    }

    void TearDown() override
    {
        test_globals::g_ucx_gotcha_gmock.reset();
        test_globals::g_comm_data_gmock.reset();
        test_globals::g_category_region_gmock.reset();
    }
};

TEST_F(ucx_gotcha_test, test_dummy)
{
    ucx_gotcha_under_test_t g;
    EXPECT_EQ(g.label(), "ucx_gotcha");
    EXPECT_EQ(g.gotcha_capacity, 100);
}

TEST_F(ucx_gotcha_test, test_shutdown)
{
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, disable()).Times(1);

    EXPECT_NO_THROW(ucx_gotcha_under_test_t::shutdown());
}

TEST_F(ucx_gotcha_test, test_start)
{
    std::function<void()> initializer;

    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, get_is_running())
        .Times(1)
        .WillOnce(::testing::Return(false));

    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, at(::testing::Ge(0)))
        .Times(100)
        .WillRepeatedly(::testing::Return(nullptr));

    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, capacity())
        .WillRepeatedly(::testing::Return(100));

    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));

    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, start()).Times(1);

    EXPECT_NO_THROW(ucx_gotcha_under_test_t::start());

    // inside start, it's only initializing initilaizer,
    // calling to be sure that all configs are set

    ASSERT_TRUE(initializer);
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, configure(::testing::_)).Times(89);
    initializer();
}
