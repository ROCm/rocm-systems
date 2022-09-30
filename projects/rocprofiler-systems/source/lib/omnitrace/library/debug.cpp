// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/debug.hpp"
#include "library/runtime.hpp"
#include "library/state.hpp"

#include <timemory/log/color.hpp>
#include <timemory/utility/filepath.hpp>

namespace omnitrace
{
namespace debug
{
namespace
{
struct source_location_history
{
    std::array<source_location, 10> data = {};
    size_t                          size = 0;
};

auto&
get_source_location_history()
{
    static thread_local auto _v = source_location_history{};
    return _v;
}
}  // namespace

void
set_source_location(source_location&& _v)
{
    auto& _hist                             = get_source_location_history();
    auto  _idx                              = _hist.size++;
    _hist.data.at(_idx % _hist.data.size()) = _v;
}

lock::lock()
: m_lk{ tim::type_mutex<decltype(std::cerr)>(), std::defer_lock }
{
    if(!m_lk.owns_lock())
    {
        push_thread_state(ThreadState::Internal);
        m_lk.lock();
    }
}

lock::~lock()
{
    if(m_lk.owns_lock())
    {
        m_lk.unlock();
        pop_thread_state();
    }
}

FILE*
get_file()
{
    static FILE* _v = []() {
        auto&& _fname = tim::get_env<std::string>("OMNITRACE_LOG_FILE", "");
        if(!_fname.empty()) tim::log::colorized() = false;
        return (_fname.empty()) ? stderr : tim::filepath::fopen(_fname, "w");
    }();
    return _v;
}
}  // namespace debug
}  // namespace omnitrace
