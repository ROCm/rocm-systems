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

#include "library/components/exit_gotcha.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"
#include "library/runtime.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/process/threading.hpp>
#include <timemory/utility/types.hpp>

#include <cstddef>
#include <cstdlib>

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

    if(get_state() < State::Finalized)
    {
        if(config::settings_are_configured())
        {
            ROCPROFSYS_VERBOSE(0, "finalizing %s before calling %s(%s)...\n",
                               get_exe_name().c_str(), _data.tool_id.c_str(),
                               JOIN(", ", _args...).c_str());
        }
        else
        {
            ROCPROFSYS_BASIC_VERBOSE(0, "finalizing %s before calling %s(%s)...\n",
                                     get_exe_name().c_str(), _data.tool_id.c_str(),
                                     JOIN(", ", _args...).c_str());
        }

        rocprofsys_finalize();
    }

    if(config::settings_are_configured())
    {
        ROCPROFSYS_VERBOSE(0, "calling %s(%s) in %s...\n", _data.tool_id.c_str(),
                           JOIN(", ", _args...).c_str(), get_exe_name().c_str());
    }
    else
    {
        ROCPROFSYS_BASIC_VERBOSE(0, "calling %s(%s) in %s...\n", _data.tool_id.c_str(),
                                 JOIN(", ", _args...).c_str(), get_exe_name().c_str());
    }

    if(_exit_info.is_known && _exit_info.exit_code != 0)
    {
        ROCPROFSYS_BASIC_VERBOSE(0, "%s exiting with non-zero exit code: %i...\n",
                                 get_exe_name().c_str(), _exit_info.exit_code);
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
