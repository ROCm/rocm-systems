// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/components/exit_gotcha.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"
#include "library/runtime.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/process/threading.hpp>
#include <timemory/utility/types.hpp>

#include "logger/debug.hpp"

#include <spdlog/fmt/ranges.h>

#include <cstddef>
#include <cstdlib>
#include <tuple>
#include <unistd.h>

namespace rocprofsys
{
namespace component
{
void
exit_gotcha::configure()
{
    exit_gotcha_t::get_initializer() = []() {
        exit_gotcha_t::configure<0, void>("abort");
        exit_gotcha_t::configure<1, void, int>("exit");
        exit_gotcha_t::configure<2, void, int>("quick_exit");
        exit_gotcha_t::configure<3, void, int>("_Exit");
    };
}

namespace
{
auto _exit_info = exit_gotcha::exit_info{};

template <typename FuncT, typename... Args>
void
invoke_exit_gotcha(const exit_gotcha::gotcha_data& _data, FuncT _func, Args... _args)
{
    threading::clear_callbacks();

    if(get_state() < State::Finalized && !is_child_process())
    {
        LOG_DEBUG("Finalizing {} before calling {}({})...", get_exe_name(), _data.tool_id,
                  fmt::join(std::forward_as_tuple(_args...), ", "));

        rocprofsys_finalize();
    }

    LOG_DEBUG("Calling {}({}) in {}...", _data.tool_id,
              fmt::join(std::forward_as_tuple(_args...), ", "), get_exe_name().c_str());

    if(_exit_info.is_known && _exit_info.exit_code != 0)
    {
        LOG_DEBUG("{} exiting with non-zero exit code: {}...", get_exe_name(),
                  _exit_info.exit_code);
    }

    (*_func)(_args...);
}
}  // namespace

// exit
// quick_exit
void
exit_gotcha::operator()(const gotcha_data& _data, exit_func_t _func, int _ec) const
{
    _exit_info = { true, _data.tool_id.find("quick") != std::string::npos, _ec };
    invoke_exit_gotcha(_data, _func, _ec);
}

// abort
void
exit_gotcha::operator()(const gotcha_data& _data, abort_func_t _func) const
{
    _exit_info = { true, false, SIGABRT };
    invoke_exit_gotcha(_data, _func);
}

exit_gotcha::exit_info
exit_gotcha::get_exit_info()
{
    return _exit_info;
}
}  // namespace component
}  // namespace rocprofsys
