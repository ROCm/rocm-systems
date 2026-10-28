#include <gtest/gtest.h>

#include <cerrno>
#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>

#include "cpu_snap/error.hpp"

namespace
{

template <typename Error>
void expect_catchable_as_every_base(const Error& original)
{
    try
    {
        throw original;
    } catch(const cpu_snap::error& e)
    {
        EXPECT_STREQ(e.what(), original.what());
    } catch(...)
    {
        ADD_FAILURE() << "not catchable as cpu_snap::error";
    }
    try
    {
        throw original;
    } catch(const std::runtime_error& e)
    {
        EXPECT_STREQ(e.what(), original.what());
    } catch(...)
    {
        ADD_FAILURE() << "not catchable as std::runtime_error";
    }
    try
    {
        throw original;
    } catch(const std::exception& e)
    {
        EXPECT_STREQ(e.what(), original.what());
    } catch(...)
    {
        ADD_FAILURE() << "not catchable as std::exception";
    }
}

}  // namespace

TEST(error_hierarchy, config_error_is_catchable_as_every_base)
{
    expect_catchable_as_every_base(cpu_snap::config_error("bad config"));
}

TEST(error_hierarchy, usage_error_is_catchable_as_every_base)
{
    expect_catchable_as_every_base(cpu_snap::usage_error("bad order"));
}

TEST(error_hierarchy, environment_error_is_catchable_as_every_base)
{
    expect_catchable_as_every_base(cpu_snap::environment_error("denied"));
}

TEST(error_hierarchy, resource_error_is_catchable_as_every_base)
{
    expect_catchable_as_every_base(cpu_snap::resource_error("full"));
}

TEST(error_hierarchy, categories_are_distinct_types)
{
    EXPECT_THROW(throw cpu_snap::config_error("x"), cpu_snap::config_error);
    try
    {
        throw cpu_snap::config_error("x");
    } catch(const cpu_snap::environment_error&)
    {
        ADD_FAILURE() << "config_error must not match environment_error";
    } catch(const cpu_snap::error&)
    {
    }
}

TEST(error_hierarchy, environment_error_carries_code_and_verbatim_text)
{
    const cpu_snap::environment_error e(std::error_code(EACCES, std::generic_category()),
                                        "perf denied");
    EXPECT_EQ(e.code(), std::error_code(EACCES, std::generic_category()));
    EXPECT_STREQ(e.what(), "perf denied");
}

TEST(error_hierarchy, environment_error_code_is_empty_by_default)
{
    const cpu_snap::environment_error e("no errno");
    EXPECT_FALSE(e.code());
    EXPECT_STREQ(e.what(), "no errno");
}

TEST(error_hierarchy, environment_error_from_errno_appends_strerror_once)
{
    const auto e = cpu_snap::environment_error::from_errno(ENOENT, "open");
    EXPECT_EQ(e.code().value(), ENOENT);
    const std::string what = e.what();
    EXPECT_EQ(what.rfind("open: ", 0), 0U);
    EXPECT_EQ(what.find(':', 6), std::string::npos);
}

TEST(error_hierarchy, resource_error_carries_code)
{
    const cpu_snap::resource_error e(std::error_code(ENOSPC, std::generic_category()),
                                     "table full");
    EXPECT_EQ(e.code(), std::errc::no_space_on_device);
    EXPECT_STREQ(e.what(), "table full");
    EXPECT_FALSE(cpu_snap::resource_error("no code").code());
}
