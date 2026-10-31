// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/env_op.hpp"
#include "delivery.hpp"
#include "system.hpp"

#include <fmt/format.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::common_utils
{
// Defined in common_utils.cpp. Forward-declared so this header does not include
// the preset and JSON headers.
[[nodiscard]] std::vector<char*>
to_c_argv(std::vector<std::string>& src);
}  // namespace rocprofsys::common_utils

namespace rocprofsys::rocsys
{
namespace detail
{
constexpr int k_exec_failure_status = 127;

[[nodiscard]] inline std::string
sibling_path(std::string_view directory, std::string_view binary)
{
    std::string path;
    path.reserve(directory.size() + binary.size() + 1);
    path.append(directory);
    if(!directory.empty() && !directory.ends_with('/'))
    {
        path.push_back('/');
    }
    path.append(binary);
    return path;
}

}  // namespace detail

/// @brief Applies a @ref launch_plan. This is the only stage that changes the process.
template <system_ops System = default_system>
class launcher
{
public:
    explicit launcher(System system = {})
    : m_system(std::move(system))
    {}

    /// @brief Run @p plan. Usage failures return 1. A failed exec returns 127.
    [[nodiscard]] int launch(const launch_plan& plan, std::string_view invocation)
    {
        if(plan.in_process)
        {
            return launch_in_process(plan, invocation);
        }
        return launch_exec(plan, invocation);
    }

private:
    [[nodiscard]] int launch_in_process(const launch_plan& plan,
                                        std::string_view   invocation)
    {
        if(!apply_env(plan.env, invocation))
        {
            return EXIT_FAILURE;
        }
        if(!plan.mode.has_value())
        {
            report(fmt::format("{}: in-process command has no tool mode\n", invocation));
            return EXIT_FAILURE;
        }
        auto argv     = plan.argv;
        auto pointers = common_utils::to_c_argv(argv);
        return m_system.run_tool(static_cast<int>(argv.size()), pointers.data(),
                                 *plan.mode);
    }

    [[nodiscard]] int launch_exec(const launch_plan& plan, std::string_view invocation)
    {
        const auto directory = m_system.executable_directory();
        if(directory.empty())
        {
            report(fmt::format("{}: failed to resolve install directory\n", invocation));
            return detail::k_exec_failure_status;
        }

        const auto path = detail::sibling_path(directory, plan.program);
        if(!m_system.exists(path.c_str()))
        {
            report(fmt::format(
                "{}: '{}' was not found next to this executable ({})\n"
                "hint: install the rocprofiler-systems tools in the same directory "
                "as rocsys\n",
                invocation, plan.program, path));
            return detail::k_exec_failure_status;
        }

        auto      argv       = plan.argv;
        auto      env        = merge_environ(m_system.environ(), plan.env);
        auto      c_argv     = common_utils::to_c_argv(argv);
        auto      c_env      = common_utils::to_c_argv(env);
        const int exec_error = m_system.exec(path.c_str(), c_argv.data(), c_env.data());
        report(fmt::format("{}: failed to execute '{}': {}\n", invocation, path,
                           std::strerror(exec_error)));
        return detail::k_exec_failure_status;
    }

    [[nodiscard]] bool apply_env(const env_plan& ops, std::string_view invocation)
    {
        for(const auto& op : ops)
        {
            if(!apply_one(op, invocation))
            {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool apply_one(const env_op& op, std::string_view invocation)
    {
        if(op.mode != env_mode::replace)
        {
            report(fmt::format("{}: unsupported environment operation for '{}'\n",
                               invocation, op.key));
            return false;
        }
        if(m_system.set_env(op.key.c_str(), op.value.c_str()) != 0)
        {
            report(fmt::format("{}: failed to set environment variable '{}'\n",
                               invocation, op.key));
            return false;
        }
        return true;
    }

    void report(std::string_view message) const { m_system.write_error(message); }

    System m_system;
};

}  // namespace rocprofsys::rocsys
