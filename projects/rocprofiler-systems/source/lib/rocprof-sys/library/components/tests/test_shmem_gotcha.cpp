// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys/library/components/shmem_gotcha.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <set>
#include <string>
#include <vector>

constexpr auto NUMBER_OF_FUNCTIONS = 101;
constexpr auto GOTCHA_CAPACITY     = 120u;

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
    MOCK_METHOD(void, stop_generic, (std::string_view name));
    MOCK_METHOD(void, stop_ptr, (std::string_view name, void* ret));
    MOCK_METHOD(void, stop_int, (std::string_view name, int ret));
    MOCK_METHOD(void, stop_long, (std::string_view name, long ret));
};

namespace test_globals
{
std::unique_ptr<GMockSHMEMGotcha>    g_shmem_gotcha_gmock;
std::unique_ptr<GMockCommData>       g_comm_data_gmock;
std::unique_ptr<GMockCategoryRegion> g_category_region_gmock;
}  // namespace test_globals

struct MockedSHMEMGotcha
{
    static bool is_permitted(const std::string& func_name)
    {
        auto& reject_fn = get_reject_list();
        if(reject_fn && reject_fn().count(func_name) > 0) return false;

        auto& permit_fn = get_permit_list();
        if(permit_fn)
        {
            const auto& permit = permit_fn();
            if(!permit.empty() && permit.count(func_name) == 0) return false;
        }
        return true;
    }

    template <int N, typename... Args>
    static void configure(std::string func_name)
    {
        if(!is_permitted(func_name)) return;
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

    static void stop(std::string_view name)
    {
        test_globals::g_category_region_gmock->stop_generic(name);
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

    static void stop(std::string_view name, const char*, long ret)
    {
        test_globals::g_category_region_gmock->stop_long(name, ret);
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
        MockedSHMEMGotcha::get_reject_list()  = []() { return std::set<std::string>{}; };
        MockedSHMEMGotcha::get_permit_list()  = []() { return std::set<std::string>{}; };
    }

    void TearDown() override
    {
        unsetenv("ROCPROFSYS_SHMEM_REJECT_LIST");
        unsetenv("ROCPROFSYS_SHMEM_PERMIT_LIST");
        test_globals::g_shmem_gotcha_gmock.reset();
        test_globals::g_comm_data_gmock.reset();
        test_globals::g_category_region_gmock.reset();
    }
};

TEST_F(shmem_gotcha_test, test_static_labels)
{
    shmem_gotcha_under_test_t g;
    EXPECT_EQ(g.label(), "shmem_gotcha");
    EXPECT_EQ(g.gotcha_capacity, GOTCHA_CAPACITY);
}

TEST_F(shmem_gotcha_test, test_component_lifecycle)
{
    setenv("ROCPROFSYS_SHMEM_REJECT_LIST", "", 1);
    setenv("ROCPROFSYS_SHMEM_PERMIT_LIST", "all", 1);
    std::function<void()> initializer;

    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, get_is_running())
        .Times(1)
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, capacity())
        .WillRepeatedly(::testing::Return(GOTCHA_CAPACITY));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, at(::testing::_))
        .WillRepeatedly(::testing::Return(nullptr));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_comm_data_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, configure(::testing::_))
        .Times(NUMBER_OF_FUNCTIONS);

    EXPECT_NO_THROW(shmem_gotcha_under_test_t::start());
    ASSERT_TRUE(initializer);
    initializer();

    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, disable()).Times(1);
    EXPECT_NO_THROW(shmem_gotcha_under_test_t::shutdown());
}

TEST_F(shmem_gotcha_test, test_shutdown)
{
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, disable()).Times(1);

    EXPECT_NO_THROW(shmem_gotcha_under_test_t::shutdown());
}

TEST_F(shmem_gotcha_test, test_start_when_already_running)
{
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, get_is_running())
        .Times(1)
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*test_globals::g_comm_data_gmock, start()).Times(0);
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, start()).Times(0);
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, configure(::testing::_)).Times(0);

    EXPECT_NO_THROW(shmem_gotcha_under_test_t::start());
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

TEST_F(shmem_gotcha_test, test_audit_outgoing_long)
{
    MockedGotchaData data;
    data.tool_id = "shmem_fadd64";

    long ret = 999999L;

    EXPECT_CALL(*test_globals::g_category_region_gmock, stop_long)
        .Times(1)
        .WillOnce([&](std::string_view name, long r) {
            EXPECT_EQ(name, "shmem_fadd64");
            EXPECT_EQ(r, 999999L);
        });

    shmem_gotcha_under_test_t::audit(data, tim::audit::outgoing{}, ret);
}

TEST_F(shmem_gotcha_test, test_audit_outgoing_no_return)
{
    MockedGotchaData data;
    data.tool_id = "shmem_barrier_all";

    EXPECT_CALL(*test_globals::g_category_region_gmock, stop_generic)
        .Times(1)
        .WillOnce([](std::string_view name) { EXPECT_EQ(name, "shmem_barrier_all"); });

    shmem_gotcha_under_test_t::audit(data, tim::audit::outgoing{});
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

TEST_F(shmem_gotcha_test, test_get_category_map)
{
    using namespace rocprofsys::component::shmem_categories;
    const auto& m = get_category_map();

    EXPECT_EQ(m.size(), 7u);
    EXPECT_NE(m.find("init"), m.end());
    EXPECT_NE(m.find("sync"), m.end());
    EXPECT_NE(m.find("rma"), m.end());
    EXPECT_NE(m.find("collective"), m.end());
    EXPECT_NE(m.find("reduction"), m.end());
    EXPECT_NE(m.find("atomics"), m.end());
    EXPECT_NE(m.find("memory"), m.end());

    size_t total = 0;
    for(const auto& kv : m)
        total += kv.second.size();
    EXPECT_EQ(total, static_cast<size_t>(NUMBER_OF_FUNCTIONS));

    EXPECT_NE(m.at("init").count("shmem_init"), 0u);
    EXPECT_NE(m.at("init").count("shmem_finalize"), 0u);
    EXPECT_NE(m.at("sync").count("shmem_barrier_all"), 0u);
    EXPECT_NE(m.at("atomics").count("shmem_fetch_and_add64"), 0u);
    EXPECT_NE(m.at("memory").count("shmem_malloc"), 0u);
}

TEST_F(shmem_gotcha_test, test_get_default_permit)
{
    using namespace rocprofsys::component::shmem_categories;
    auto permit = get_default_permit();

    EXPECT_NE(permit.count("shmem_init"), 0u);
    EXPECT_NE(permit.count("shmem_barrier_all"), 0u);
    EXPECT_NE(permit.count("shmem_put32"), 0u);

    auto atomics = get_category_map().at("atomics");
    for(const auto& api : atomics)
        EXPECT_EQ(permit.count(api), 0u)
            << "atomics should be excluded from default permit: " << api;

    auto memory = get_category_map().at("memory");
    for(const auto& api : memory)
        EXPECT_EQ(permit.count(api), 0u)
            << "memory should be excluded from default permit: " << api;
}

TEST_F(shmem_gotcha_test, test_expand_tokens_to_apis)
{
    using namespace rocprofsys::component::shmem_categories;
    const auto& m = get_category_map();

    std::set<std::string> init_only = { "init" };
    auto                  expanded  = expand_tokens_to_apis(init_only);
    EXPECT_EQ(expanded, m.at("init"));

    std::set<std::string> raw_api = { "shmem_init" };
    EXPECT_EQ(expand_tokens_to_apis(raw_api), std::set<std::string>{ "shmem_init" });

    std::set<std::string> mixed          = { "init", "shmem_malloc" };
    auto                  mixed_expanded = expand_tokens_to_apis(mixed);
    EXPECT_EQ(mixed_expanded.count("shmem_init"), 1u);
    EXPECT_EQ(mixed_expanded.count("shmem_malloc"), 1u);
}

TEST_F(shmem_gotcha_test, test_configure_function_names)
{
    unsetenv("ROCPROFSYS_SHMEM_REJECT_LIST");
    setenv("ROCPROFSYS_SHMEM_PERMIT_LIST", "all", 1);
    std::function<void()>    initializer;
    std::vector<std::string> configured_names;

    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, get_is_running())
        .Times(1)
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, capacity())
        .WillRepeatedly(::testing::Return(GOTCHA_CAPACITY));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, at(::testing::_))
        .WillRepeatedly(::testing::Return(nullptr));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, get_initializer())
        .Times(1)
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_comm_data_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, configure(::testing::_))
        .Times(NUMBER_OF_FUNCTIONS)
        .WillRepeatedly([&configured_names](std::string name) {
            configured_names.push_back(std::move(name));
        });

    shmem_gotcha_under_test_t::start();
    ASSERT_TRUE(initializer);
    initializer();

    ASSERT_EQ(configured_names.size(), static_cast<size_t>(NUMBER_OF_FUNCTIONS));

    std::set<std::string> expected_names;
    for(const auto& kv : rocprofsys::component::shmem_categories::get_category_map())
    {
        for(const auto& name : kv.second)
            expected_names.insert(name);
    }
    EXPECT_EQ(expected_names.size(), static_cast<size_t>(NUMBER_OF_FUNCTIONS));

    std::set<std::string> configured_set(configured_names.begin(),
                                         configured_names.end());
    EXPECT_EQ(configured_set, expected_names);
}

TEST_F(shmem_gotcha_test, test_get_reject_list_assignable_and_invokable)
{
    auto reject = std::set<std::string>{ "shmem_init", "shmem_finalize" };
    MockedSHMEMGotcha::get_reject_list() = [reject]() { return reject; };
    EXPECT_EQ(MockedSHMEMGotcha::get_reject_list()(), reject);
}

TEST_F(shmem_gotcha_test, test_reject_list_excludes_from_configure)
{
    setenv("ROCPROFSYS_SHMEM_REJECT_LIST", "shmem_init,shmem_finalize", 1);
    setenv("ROCPROFSYS_SHMEM_PERMIT_LIST", "all", 1);

    std::function<void()>    initializer;
    std::vector<std::string> configured_names;

    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, get_is_running())
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, capacity())
        .WillRepeatedly(::testing::Return(GOTCHA_CAPACITY));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, at(::testing::_))
        .WillRepeatedly(::testing::Return(nullptr));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, get_initializer())
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_comm_data_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, configure(::testing::_))
        .WillRepeatedly([&configured_names](std::string name) {
            configured_names.push_back(std::move(name));
        });

    shmem_gotcha_under_test_t::start();
    ASSERT_TRUE(initializer);
    initializer();

    EXPECT_EQ(configured_names.end(),
              std::find(configured_names.begin(), configured_names.end(), "shmem_init"));
    EXPECT_EQ(
        configured_names.end(),
        std::find(configured_names.begin(), configured_names.end(), "shmem_finalize"));
    EXPECT_EQ(configured_names.size(), static_cast<size_t>(NUMBER_OF_FUNCTIONS - 2));

    unsetenv("ROCPROFSYS_SHMEM_REJECT_LIST");
    unsetenv("ROCPROFSYS_SHMEM_PERMIT_LIST");
}

TEST_F(shmem_gotcha_test, test_permit_list_restricts_configure)
{
    setenv("ROCPROFSYS_SHMEM_REJECT_LIST", "", 1);
    setenv("ROCPROFSYS_SHMEM_PERMIT_LIST", "shmem_put32,shmem_get32", 1);

    std::function<void()>    initializer;
    std::vector<std::string> configured_names;

    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, get_is_running())
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, capacity())
        .WillRepeatedly(::testing::Return(GOTCHA_CAPACITY));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, at(::testing::_))
        .WillRepeatedly(::testing::Return(nullptr));
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, get_initializer())
        .WillOnce(::testing::ReturnRef(initializer));
    EXPECT_CALL(*test_globals::g_comm_data_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, start()).Times(1);
    EXPECT_CALL(*test_globals::g_shmem_gotcha_gmock, configure(::testing::_))
        .WillRepeatedly([&configured_names](std::string name) {
            configured_names.push_back(std::move(name));
        });

    shmem_gotcha_under_test_t::start();
    ASSERT_TRUE(initializer);
    initializer();

    EXPECT_EQ(configured_names.size(), 2u);
    EXPECT_NE(configured_names.end(),
              std::find(configured_names.begin(), configured_names.end(), "shmem_put32"));
    EXPECT_NE(configured_names.end(),
              std::find(configured_names.begin(), configured_names.end(), "shmem_get32"));

    unsetenv("ROCPROFSYS_SHMEM_REJECT_LIST");
    unsetenv("ROCPROFSYS_SHMEM_PERMIT_LIST");
}

TEST_F(shmem_gotcha_test, test_get_permit_list_assignable_and_invokable)
{
    auto permit = std::set<std::string>{ "shmem_put32", "shmem_get32" };
    MockedSHMEMGotcha::get_permit_list() = [permit]() { return permit; };
    EXPECT_EQ(MockedSHMEMGotcha::get_permit_list()(), permit);
}

TEST_F(shmem_gotcha_test, test_different_gotcha_tool_ids)
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
