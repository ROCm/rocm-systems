// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys/library/components/ucx_gotcha.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace
{
constexpr auto NUMBER_OF_FUNCTIONS = 93;
constexpr auto GOTCHA_CAPACITY     = 100u;

struct MockedGotchaData
{
    std::string tool_id;
    int         verbose = 0;  // set by ucx_gotcha::configure() to silence gotcha output
};

struct GMockUCXGotcha
{
    MOCK_METHOD(void, configure, (std::string func_name));
    MOCK_METHOD(size_t, capacity, ());
    MOCK_METHOD(void*, at, (size_t index));
    MOCK_METHOD(void, disable, ());
    MOCK_METHOD(bool, get_is_running, (), (const));
    MOCK_METHOD(void, start, ());
    MOCK_METHOD(std::function<void()>&, get_initializer, ());
    MOCK_METHOD(void, set_ready, (bool ready));
};

struct GMockCommData
{
    MOCK_METHOD(void, start, ());
    MOCK_METHOD(void, audit, ());
};

struct GMockCategoryRegion
{
    MOCK_METHOD(void, start_generic, (std::string_view name));
    MOCK_METHOD(void, stop_generic, (std::string_view name));
    MOCK_METHOD(void, stop_ptr, (std::string_view name, void* ret));
    MOCK_METHOD(void, stop_int, (std::string_view name, int ret));
    MOCK_METHOD(void, stop_unsigned, (std::string_view name, unsigned ret));
    MOCK_METHOD(void, stop_long, (std::string_view name, long ret));
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
        test_globals::g_ucx_gotcha_gmock->configure(std::move(func_name));
    }
    static size_t capacity() { return test_globals::g_ucx_gotcha_gmock->capacity(); }
    static void*  at(size_t index) { return test_globals::g_ucx_gotcha_gmock->at(index); }
    static void   disable() { test_globals::g_ucx_gotcha_gmock->disable(); }
    bool          get_is_running() const
    {
        return test_globals::g_ucx_gotcha_gmock->get_is_running();
    }
    void                          start() { test_globals::g_ucx_gotcha_gmock->start(); }
    static std::function<void()>& get_initializer()
    {
        return test_globals::g_ucx_gotcha_gmock->get_initializer();
    }
    static void set_ready(bool ready)
    {
        test_globals::g_ucx_gotcha_gmock->set_ready(ready);
    }
};

template <typename GotchaData>
struct MockedCommData
{
    static void start() { test_globals::g_comm_data_gmock->start(); }

    template <typename... Args>
    static void audit(Args&&...)
    {
        test_globals::g_comm_data_gmock->audit();
    }
};

struct MockedCategoryRegion
{
    template <typename... Args>
    static void start(std::string_view name, Args&&...)
    {
        test_globals::g_category_region_gmock->start_generic(name);
    }

    static void stop(std::string_view name)
    {
        test_globals::g_category_region_gmock->stop_generic(name);
    }

    template <typename... Args>
    static void stop(std::string_view, Args&&...)
    {
        FAIL() << "Unexpected call of category_region::stop";
    }

    static void stop(std::string_view name, const char*, void* ret)
    {
        test_globals::g_category_region_gmock->stop_ptr(name, ret);
    }

    static void stop(std::string_view name, const char*, int ret)
    {
        test_globals::g_category_region_gmock->stop_int(name, ret);
    }

    static void stop(std::string_view name, const char*, unsigned ret)
    {
        test_globals::g_category_region_gmock->stop_unsigned(name, ret);
    }

    static void stop(std::string_view name, const char*, long ret)
    {
        test_globals::g_category_region_gmock->stop_long(name, ret);
    }
};

struct MockGotchaBundle
{
    MockedUCXGotcha instance;

    template <typename>
    MockedUCXGotcha* get()
    {
        return &instance;
    }
};

struct MockedUCXPolicy
{
    using gotcha_data     = MockedGotchaData;
    using comm_data       = MockedCommData<gotcha_data>;
    using category_region = MockedCategoryRegion;
    using ucx_bundle_t    = void;
    using ucx_gotcha_t    = MockedUCXGotcha;
    using gotcha_bundle_t = MockGotchaBundle;
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

TEST_F(ucx_gotcha_test, test_static_labels)
{
    ucx_gotcha_under_test_t g;
    EXPECT_EQ(g.label(), "ucx_gotcha");
    EXPECT_EQ(g.gotcha_capacity, GOTCHA_CAPACITY);
}

TEST_F(ucx_gotcha_test, test_component_lifecycle)
{
    std::function<void()> initializer;

    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, get_is_running())
        .Times(1)
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, capacity())
        .WillRepeatedly(::testing::Return(GOTCHA_CAPACITY));
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, at(::testing::_))
        .WillRepeatedly(::testing::Return(nullptr));
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_comm_data_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, configure(::testing::_))
        .Times(NUMBER_OF_FUNCTIONS);

    EXPECT_NO_THROW(ucx_gotcha_under_test_t::start());
    ASSERT_TRUE(initializer);
    initializer();

    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, disable()).Times(1);
    EXPECT_NO_THROW(ucx_gotcha_under_test_t::shutdown());
}

TEST_F(ucx_gotcha_test, test_shutdown)
{
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, disable()).Times(1);

    EXPECT_NO_THROW(ucx_gotcha_under_test_t::shutdown());
}

TEST_F(ucx_gotcha_test, test_start_when_already_running)
{
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, get_is_running())
        .Times(1)
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*test_globals::g_comm_data_gmock, start()).Times(0);
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, start()).Times(0);
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, configure(::testing::_)).Times(0);

    EXPECT_NO_THROW(ucx_gotcha_under_test_t::start());
}

TEST_F(ucx_gotcha_test, test_configure_function_names)
{
    std::function<void()>    initializer;
    std::vector<std::string> configured_names;

    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, get_is_running())
        .Times(1)
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, capacity())
        .WillRepeatedly(::testing::Return(GOTCHA_CAPACITY));
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, at(::testing::_))
        .WillRepeatedly(::testing::Return(nullptr));
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_comm_data_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, configure(::testing::_))
        .Times(NUMBER_OF_FUNCTIONS)
        .WillRepeatedly([&configured_names](std::string name) {
            configured_names.push_back(std::move(name));
        });

    ucx_gotcha_under_test_t::start();
    ASSERT_TRUE(initializer);
    initializer();

    ASSERT_EQ(configured_names.size(), static_cast<size_t>(NUMBER_OF_FUNCTIONS));
    // A representative sample of the wrapped UCX symbols.
    EXPECT_NE(
        configured_names.end(),
        std::find(configured_names.begin(), configured_names.end(), "ucp_tag_send_nbx"));
    EXPECT_NE(
        configured_names.end(),
        std::find(configured_names.begin(), configured_names.end(), "ucp_tag_recv_nbx"));
    EXPECT_NE(configured_names.end(),
              std::find(configured_names.begin(), configured_names.end(), "ucp_put_nbx"));
    EXPECT_NE(configured_names.end(),
              std::find(configured_names.begin(), configured_names.end(), "ucp_get_nbx"));
}

TEST_F(ucx_gotcha_test, test_audit_incoming_generic)
{
    MockedGotchaData data;
    data.tool_id = "ucp_worker_progress";

    EXPECT_CALL(*test_globals::g_category_region_gmock, start_generic)
        .Times(1)
        .WillOnce([](std::string_view name) { EXPECT_EQ(name, "ucp_worker_progress"); });

    ucx_gotcha_under_test_t::audit(data, tim::audit::incoming{});
}

TEST_F(ucx_gotcha_test, test_audit_outgoing_no_return)
{
    MockedGotchaData data;
    data.tool_id = "ucp_tag_send_nbx";

    // Outgoing audit with no return value must still call stop so the region is closed.
    EXPECT_CALL(*test_globals::g_category_region_gmock, stop_generic)
        .Times(1)
        .WillOnce([](std::string_view name) { EXPECT_EQ(name, "ucp_tag_send_nbx"); });

    ucx_gotcha_under_test_t::audit(data, tim::audit::outgoing{});
}

TEST_F(ucx_gotcha_test, test_audit_outgoing_ptr)
{
    MockedGotchaData data;
    data.tool_id = "ucp_put_nbx";

    void* ret = reinterpret_cast<void*>(0x1234);

    EXPECT_CALL(*test_globals::g_category_region_gmock, stop_ptr)
        .Times(1)
        .WillOnce([&](std::string_view name, void* r) {
            EXPECT_EQ(name, "ucp_put_nbx");
            EXPECT_EQ(r, ret);
        });

    ucx_gotcha_under_test_t::audit(data, tim::audit::outgoing{}, ret);
}

TEST_F(ucx_gotcha_test, test_audit_outgoing_int)
{
    MockedGotchaData data;
    data.tool_id = "ucp_put";

    int ret = 7;

    EXPECT_CALL(*test_globals::g_category_region_gmock, stop_int)
        .Times(1)
        .WillOnce([&](std::string_view name, int r) {
            EXPECT_EQ(name, "ucp_put");
            EXPECT_EQ(r, 7);
        });

    ucx_gotcha_under_test_t::audit(data, tim::audit::outgoing{}, ret);
}

TEST_F(ucx_gotcha_test, test_audit_outgoing_unsigned)
{
    MockedGotchaData data;
    data.tool_id = "ucp_worker_progress";

    // Progress APIs (e.g. ucp_worker_progress, uct_iface_progress) return unsigned
    unsigned ret = 3;

    EXPECT_CALL(*test_globals::g_category_region_gmock, stop_unsigned)
        .Times(1)
        .WillOnce([&](std::string_view name, unsigned r) {
            EXPECT_EQ(name, "ucp_worker_progress");
            EXPECT_EQ(r, 3u);
        });

    ucx_gotcha_under_test_t::audit(data, tim::audit::outgoing{}, ret);
}

TEST_F(ucx_gotcha_test, test_audit_outgoing_long)
{
    MockedGotchaData data;
    data.tool_id = "uct_ep_am_bcopy";

    // ssize_t-returning UCT APIs decay to long; the value must not be truncated to int.
    long ret = 5000000000L;

    EXPECT_CALL(*test_globals::g_category_region_gmock, stop_long)
        .Times(1)
        .WillOnce([&](std::string_view name, long r) {
            EXPECT_EQ(name, "uct_ep_am_bcopy");
            EXPECT_EQ(r, 5000000000L);
        });

    ucx_gotcha_under_test_t::audit(data, tim::audit::outgoing{}, ret);
}

TEST_F(ucx_gotcha_test, test_audit_outgoing_null_ptr)
{
    MockedGotchaData data;
    data.tool_id = "ucp_get_nbx";

    EXPECT_CALL(*test_globals::g_category_region_gmock, stop_ptr)
        .Times(1)
        .WillOnce([](std::string_view name, void* r) {
            EXPECT_EQ(name, "ucp_get_nbx");
            EXPECT_EQ(r, nullptr);
        });

    EXPECT_NO_THROW(
        ucx_gotcha_under_test_t::audit(data, tim::audit::outgoing{}, nullptr));
}

TEST_F(ucx_gotcha_test, test_audit_incoming_tag_send)
{
    MockedGotchaData data;
    data.tool_id = "ucp_tag_send_nbx";

    // The specialized incoming overload must both open the region and feed comm_data.
    EXPECT_CALL(*test_globals::g_category_region_gmock, start_generic)
        .Times(1)
        .WillOnce([](std::string_view name) { EXPECT_EQ(name, "ucp_tag_send_nbx"); });
    EXPECT_CALL(*test_globals::g_comm_data_gmock, audit()).Times(1);

    void*       ep     = reinterpret_cast<void*>(0x1);
    const void* buffer = reinterpret_cast<const void*>(0x2);
    const void* param  = reinterpret_cast<const void*>(0x3);
    ucx_gotcha_under_test_t::audit(data, tim::audit::incoming{}, ep, buffer, size_t{ 64 },
                                   std::uint64_t{ 42 }, param);
}

TEST_F(ucx_gotcha_test, test_audit_incoming_rma_put)
{
    MockedGotchaData data;
    data.tool_id = "ucp_put_nbx";

    EXPECT_CALL(*test_globals::g_category_region_gmock, start_generic)
        .Times(1)
        .WillOnce([](std::string_view name) { EXPECT_EQ(name, "ucp_put_nbx"); });
    EXPECT_CALL(*test_globals::g_comm_data_gmock, audit()).Times(1);

    void*       ep     = reinterpret_cast<void*>(0x1);
    const void* buffer = reinterpret_cast<const void*>(0x2);
    void*       rkey   = reinterpret_cast<void*>(0x3);
    const void* param  = reinterpret_cast<const void*>(0x4);
    ucx_gotcha_under_test_t::audit(data, tim::audit::incoming{}, ep, buffer,
                                   size_t{ 128 }, std::uint64_t{ 0xdead }, rkey, param);
}

TEST_F(ucx_gotcha_test, test_different_gotcha_tool_ids)
{
    auto test_incoming = [](const std::string& tool_id) {
        MockedGotchaData data;
        data.tool_id = tool_id;
        EXPECT_CALL(*test_globals::g_category_region_gmock, start_generic)
            .Times(1)
            .WillOnce([&tool_id](std::string_view name) { EXPECT_EQ(name, tool_id); });
        ucx_gotcha_under_test_t::audit(data, tim::audit::incoming{});
    };

    test_incoming("ucp_worker_progress");
    test_incoming("ucp_ep_create");
    test_incoming("ucp_mem_map");
    test_incoming("ucp_am_send_nbx");
    test_incoming("uct_ep_am_short");
}

TEST_F(ucx_gotcha_test, test_pause)
{
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, set_ready(false)).Times(1);

    EXPECT_NO_THROW(ucx_gotcha_under_test_t::pause());
}

TEST_F(ucx_gotcha_test, test_resume)
{
    EXPECT_CALL(*test_globals::g_ucx_gotcha_gmock, set_ready(true)).Times(1);

    EXPECT_NO_THROW(ucx_gotcha_under_test_t::resume());
}

TEST_F(ucx_gotcha_test, test_stop_is_noop)
{
    // stop() is intentionally a no-op
    EXPECT_NO_THROW(ucx_gotcha_under_test_t::stop());
}

// The specialized incoming overloads must each open the region (start) and forward to
// comm_data::audit. Exercising every overload guards their distinct signatures against
// accidentally collapsing into the generic template (which would skip comm_data).
TEST_F(ucx_gotcha_test, test_audit_incoming_tag_recv)
{
    MockedGotchaData data;
    data.tool_id = "ucp_tag_recv_nbx";

    EXPECT_CALL(*test_globals::g_category_region_gmock, start_generic)
        .Times(1)
        .WillOnce([](std::string_view name) { EXPECT_EQ(name, "ucp_tag_recv_nbx"); });
    EXPECT_CALL(*test_globals::g_comm_data_gmock, audit()).Times(1);

    void*       worker = reinterpret_cast<void*>(0x1);
    void*       buffer = reinterpret_cast<void*>(0x2);
    const void* param  = reinterpret_cast<const void*>(0x3);
    ucx_gotcha_under_test_t::audit(data, tim::audit::incoming{}, worker, buffer,
                                   size_t{ 64 }, std::uint64_t{ 1 }, std::uint64_t{ 2 },
                                   param);
}

TEST_F(ucx_gotcha_test, test_audit_incoming_rma_get)
{
    MockedGotchaData data;
    data.tool_id = "ucp_get_nbx";

    EXPECT_CALL(*test_globals::g_category_region_gmock, start_generic)
        .Times(1)
        .WillOnce([](std::string_view name) { EXPECT_EQ(name, "ucp_get_nbx"); });
    EXPECT_CALL(*test_globals::g_comm_data_gmock, audit()).Times(1);

    void*       ep     = reinterpret_cast<void*>(0x1);
    void*       buffer = reinterpret_cast<void*>(0x2);
    void*       rkey   = reinterpret_cast<void*>(0x3);
    const void* param  = reinterpret_cast<const void*>(0x4);
    ucx_gotcha_under_test_t::audit(data, tim::audit::incoming{}, ep, buffer,
                                   size_t{ 128 }, std::uint64_t{ 0xbeef }, rkey, param);
}

TEST_F(ucx_gotcha_test, test_audit_incoming_am_send)
{
    MockedGotchaData data;
    data.tool_id = "ucp_am_send_nbx";

    EXPECT_CALL(*test_globals::g_category_region_gmock, start_generic)
        .Times(1)
        .WillOnce([](std::string_view name) { EXPECT_EQ(name, "ucp_am_send_nbx"); });
    EXPECT_CALL(*test_globals::g_comm_data_gmock, audit()).Times(1);

    void*       ep     = reinterpret_cast<void*>(0x1);
    const void* header = reinterpret_cast<const void*>(0x2);
    const void* buffer = reinterpret_cast<const void*>(0x3);
    const void* param  = reinterpret_cast<const void*>(0x4);
    ucx_gotcha_under_test_t::audit(data, tim::audit::incoming{}, ep, unsigned{ 5 },
                                   header, size_t{ 16 }, buffer, size_t{ 256 }, param);
}

TEST_F(ucx_gotcha_test, test_audit_incoming_stream_send)
{
    MockedGotchaData data;
    data.tool_id = "ucp_stream_send_nbx";

    EXPECT_CALL(*test_globals::g_category_region_gmock, start_generic)
        .Times(1)
        .WillOnce([](std::string_view name) { EXPECT_EQ(name, "ucp_stream_send_nbx"); });
    EXPECT_CALL(*test_globals::g_comm_data_gmock, audit()).Times(1);

    void*       ep     = reinterpret_cast<void*>(0x1);
    const void* buffer = reinterpret_cast<const void*>(0x2);
    const void* param  = reinterpret_cast<const void*>(0x3);
    ucx_gotcha_under_test_t::audit(data, tim::audit::incoming{}, ep, buffer, size_t{ 32 },
                                   param);
}

TEST_F(ucx_gotcha_test, test_audit_incoming_stream_recv)
{
    MockedGotchaData data;
    data.tool_id = "ucp_stream_recv_nbx";

    EXPECT_CALL(*test_globals::g_category_region_gmock, start_generic)
        .Times(1)
        .WillOnce([](std::string_view name) { EXPECT_EQ(name, "ucp_stream_recv_nbx"); });
    EXPECT_CALL(*test_globals::g_comm_data_gmock, audit()).Times(1);

    void*       ep     = reinterpret_cast<void*>(0x1);
    void*       buffer = reinterpret_cast<void*>(0x2);
    size_t      length = 0;
    const void* param  = reinterpret_cast<const void*>(0x3);
    ucx_gotcha_under_test_t::audit(data, tim::audit::incoming{}, ep, buffer, size_t{ 32 },
                                   &length, param);
}
}  // namespace
