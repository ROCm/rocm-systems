// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys/library/components/shmem_gotcha.hpp"

#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <set>
#include <string>
#include <vector>

struct MockedGotchaData
{
    std::string tool_id;
    int         verbose = 0;  // set by shmem_gotcha::configure() to silence gotcha output
};

struct GMockSHMEMGotcha
{
    MOCK_METHOD(void, configure, (std::string func_name));
    MOCK_METHOD(size_t, capacity, ());
    MOCK_METHOD(void*, at, (size_t index));
    MOCK_METHOD(void, disable, ());
    MOCK_METHOD(bool, get_is_running, (), (const));
    MOCK_METHOD(void, start, ());
    MOCK_METHOD(std::function<void()>&, get_initializer, ());
};

struct GMockCommData
{
    MOCK_METHOD(void, start, ());
};

struct GMockCategoryRegion
{
    MOCK_METHOD(void, start_generic, (std::string_view name));
    MOCK_METHOD(void, stop_ptr, (std::string_view name, void* ret));
    MOCK_METHOD(void, stop_int, (std::string_view name, int ret));
};

namespace test_globals
{
std::unique_ptr<GMockSHMEMGotcha>    g_shmem_gotcha_gmock;
std::unique_ptr<GMockCommData>       g_comm_data_gmock;
std::unique_ptr<GMockCategoryRegion> g_category_region_gmock;
}  // namespace test_globals

struct MockedSHMEMGotcha
{
    template <int N, typename... Args>
    static void configure(std::string func_name)
    {
        test_globals::g_shmem_gotcha_gmock->configure(std::move(func_name));
    }
    static size_t capacity() { return test_globals::g_shmem_gotcha_gmock->capacity(); }
    static void*  at(size_t index)
    {
        return test_globals::g_shmem_gotcha_gmock->at(index);
    }
    static void disable() { test_globals::g_shmem_gotcha_gmock->disable(); }
    bool        get_is_running() const
    {
        return test_globals::g_shmem_gotcha_gmock->get_is_running();
    }
    void                          start() { test_globals::g_shmem_gotcha_gmock->start(); }
    static std::function<void()>& get_initializer()
    {
        return test_globals::g_shmem_gotcha_gmock->get_initializer();
    }
    static std::function<std::set<std::string>()>& get_reject_list()
    {
        static std::function<std::set<std::string>()> f;
        return f;
    }
    static std::function<std::set<std::string>()>& get_permit_list()
    {
        static std::function<std::set<std::string>()> f;
        return f;
    }
};

template <typename GotchaData>
struct MockedCommData
{
    static void start() { test_globals::g_comm_data_gmock->start(); }
};

struct MockedCategoryRegion
{
    template <typename... Args>
    static void start(std::string_view name, Args&&...)
    {
        test_globals::g_category_region_gmock->start_generic(name);
    }

    template <typename... Args>
    static void stop(std::string_view name, Args&&...)
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
};

struct MockedSHMEMPolicy
{
    using gotcha_data     = MockedGotchaData;
    using comm_data       = MockedCommData<gotcha_data>;
    using category_region = MockedCategoryRegion;
    using shmem_bundle_t  = void;
    using shmem_gotcha_t  = MockedSHMEMGotcha;
};

using shmem_gotcha_under_test_t = rocprofsys::component::shmem_gotcha<MockedSHMEMPolicy>;

class shmem_gotcha_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        test_globals::g_shmem_gotcha_gmock    = std::make_unique<GMockSHMEMGotcha>();
        test_globals::g_comm_data_gmock       = std::make_unique<GMockCommData>();
        test_globals::g_category_region_gmock = std::make_unique<GMockCategoryRegion>();
    }

    void TearDown() override
    {
        test_globals::g_shmem_gotcha_gmock.reset();
        test_globals::g_comm_data_gmock.reset();
        test_globals::g_category_region_gmock.reset();
    }
};

TEST_F(shmem_gotcha_test, static_labels)
{
    shmem_gotcha_under_test_t g;
    EXPECT_EQ(g.label(), "shmem_gotcha");
    EXPECT_EQ(g.gotcha_capacity, 180u);
}

TEST_F(shmem_gotcha_test, component_lifecycle)
{
    std::function<void()> initializer;

    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, get_is_running())
        .Times(1)
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, capacity())
        .WillRepeatedly(::testing::Return(180));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, at(::testing::_))
        .WillRepeatedly(::testing::Return(nullptr));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_comm_data_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, configure(::testing::_)).Times(101);

    EXPECT_NO_THROW(shmem_gotcha_under_test_t::start());
    ASSERT_TRUE(initializer);
    initializer();

    EXPECT_NO_THROW(shmem_gotcha_under_test_t::stop());

    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, disable()).Times(1);
    EXPECT_NO_THROW(shmem_gotcha_under_test_t::shutdown());
}

TEST_F(shmem_gotcha_test, test_shutdown)
{
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, disable()).Times(1);

    EXPECT_NO_THROW(shmem_gotcha_under_test_t::shutdown());
}

TEST_F(shmem_gotcha_test, test_stop)
{
    EXPECT_NO_THROW(shmem_gotcha_under_test_t::stop());
}

TEST_F(shmem_gotcha_test, test_start)
{
    std::function<void()> initializer;

    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, get_is_running())
        .Times(1)
        .WillOnce(::testing::Return(false));

    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, capacity())
        .WillRepeatedly(::testing::Return(180));

    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, at(::testing::_))
        .WillRepeatedly(::testing::Return(nullptr));

    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));

    EXPECT_CALL(*test_globals::g_comm_data_gmock, start()).Times(1);

    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, start()).Times(1);

    EXPECT_NO_THROW(shmem_gotcha_under_test_t::start());

    ASSERT_TRUE(initializer);
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, configure(::testing::_)).Times(101);
    initializer();
}

TEST_F(shmem_gotcha_test, test_audit_incoming_generic)
{
    MockedGotchaData data;
    data.tool_id = "shmem_put32";

    EXPECT_CALL(*test_globals::g_category_region_gmock, start_generic)
        .Times(1)
        .WillOnce([](std::string_view name) { EXPECT_EQ(name, "shmem_put32"); });

    shmem_gotcha_under_test_t::audit(data, tim::audit::incoming{});
}

TEST_F(shmem_gotcha_test, test_audit_outgoing_ptr)
{
    MockedGotchaData data;
    data.tool_id = "shmem_malloc";

    void* ret = reinterpret_cast<void*>(0x1234);

    EXPECT_CALL(*test_globals::g_category_region_gmock, stop_ptr)
        .Times(1)
        .WillOnce([&](std::string_view name, void* r) {
            EXPECT_EQ(name, "shmem_malloc");
            EXPECT_EQ(r, ret);
        });

    shmem_gotcha_under_test_t::audit(data, tim::audit::outgoing{}, ret);
}

TEST_F(shmem_gotcha_test, test_audit_outgoing_int)
{
    MockedGotchaData data;
    data.tool_id = "shmem_my_pe";

    int ret = 42;

    EXPECT_CALL(*test_globals::g_category_region_gmock, stop_int)
        .Times(1)
        .WillOnce([&](std::string_view name, int r) {
            EXPECT_EQ(name, "shmem_my_pe");
            EXPECT_EQ(r, 42);
        });

    shmem_gotcha_under_test_t::audit(data, tim::audit::outgoing{}, ret);
}

TEST_F(shmem_gotcha_test, test_audit_incoming_empty_tool_id)
{
    MockedGotchaData data;
    data.tool_id = "";

    EXPECT_CALL(*test_globals::g_category_region_gmock, start_generic)
        .Times(1)
        .WillOnce([](std::string_view name) { EXPECT_EQ(name, ""); });

    EXPECT_NO_THROW(shmem_gotcha_under_test_t::audit(data, tim::audit::incoming{}));
}

TEST_F(shmem_gotcha_test, test_audit_outgoing_null_ptr)
{
    MockedGotchaData data;
    data.tool_id = "shmem_malloc";

    EXPECT_CALL(*test_globals::g_category_region_gmock, stop_ptr)
        .Times(1)
        .WillOnce([](std::string_view name, void* r) {
            EXPECT_EQ(name, "shmem_malloc");
            EXPECT_EQ(r, nullptr);
        });

    EXPECT_NO_THROW(
        shmem_gotcha_under_test_t::audit(data, tim::audit::outgoing{}, nullptr));
}

TEST_F(shmem_gotcha_test, different_gotcha_tool_ids)
{
    auto test_incoming = [this](const std::string& tool_id) {
        MockedGotchaData data;
        data.tool_id = tool_id;
        EXPECT_CALL(*test_globals::g_category_region_gmock, start_generic)
            .Times(1)
            .WillOnce([&tool_id](std::string_view name) { EXPECT_EQ(name, tool_id); });
        shmem_gotcha_under_test_t::audit(data, tim::audit::incoming{});
    };

    test_incoming("shmem_init");
    test_incoming("shmem_finalize");
    test_incoming("shmem_barrier_all");
    test_incoming("shmem_put32");
    test_incoming("shmem_get64");
    test_incoming("shmem_malloc");
    test_incoming("shmem_my_pe");
    test_incoming("shmem_fadd64");
}

TEST_F(shmem_gotcha_test, test_configure_function_names)
{
    std::function<void()>    initializer;
    std::vector<std::string> configured_names;

    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, get_is_running())
        .Times(1)
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, capacity())
        .WillRepeatedly(::testing::Return(180));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, at(::testing::_))
        .WillRepeatedly(::testing::Return(nullptr));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_comm_data_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, configure(::testing::_))
        .Times(101)
        .WillRepeatedly([&configured_names](std::string name) {
            configured_names.push_back(std::move(name));
        });

    shmem_gotcha_under_test_t::start();
    ASSERT_TRUE(initializer);
    initializer();

    ASSERT_GE(configured_names.size(), 2u);
    EXPECT_EQ(configured_names.front(), "shmem_init");
    EXPECT_EQ(configured_names.back(), "shmem_fetch_and_add64");
}
