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

#include "debug.hpp"
#include "binary/address_range.hpp"
#include "locking.hpp"
#include "state.hpp"

#include <timemory/log/color.hpp>
#include <timemory/process/threading.hpp>
#include <timemory/utility/filepath.hpp>

#include <iomanip>
#include <sstream>
#include <string>

namespace rocprofsys
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

const std::string&
get_file_name()
{
    static auto _fname = tim::get_env<std::string>("ROCPROFSYS_LOG_FILE", "");
    return _fname;
}

std::atomic<FILE*>&
get_file_pointer()
{
    static auto _v = std::atomic<FILE*>{ []() {
        const auto& _fname = get_file_name();
        if(!_fname.empty()) tim::log::monochrome() = true;
        return (_fname.empty())
                   ? stderr
                   : filepath::fopen(
                         settings::format(_fname, filepath::basename(filepath::realpath(
                                                      "/proc/self/exe", nullptr, false))),
                         "w");
    }() };
    return _v;
}

auto&
get_source_location_history()
{
    static thread_local auto _v = source_location_history{};
    return _v;
}

auto _protect_lock   = std::atomic<bool>{ false };
auto _protect_unlock = std::atomic<bool>{ false };
}  // namespace

void
set_source_location(source_location&& _v)
{
    auto& _hist                             = get_source_location_history();
    auto  _idx                              = _hist.size++;
    _hist.data.at(_idx % _hist.data.size()) = _v;
}

lock::lock()
: m_lk{ tim::type_mutex<decltype(std::cerr), TIMEMORY_API, 1, locking::atomic_mutex>(),
        std::defer_lock }
{
    if(!m_lk.owns_lock() && !_protect_lock)
    {
        _protect_lock.store(true);
        push_thread_state(ThreadState::Internal);
        m_lk.lock();
        _protect_lock.store(false);
    }
}

lock::~lock()
{
    if(m_lk.owns_lock() && !_protect_unlock)
    {
        _protect_unlock.store(true);
        m_lk.unlock();
        pop_thread_state();
        _protect_unlock.store(false);
    }
}

FILE*
get_file()
{
    return get_file_pointer();
}

void
close_file()
{
    if(get_file() != stderr)
    {
        auto* _file = get_file_pointer().load();
        get_file_pointer().store(stderr);
        fclose(_file);
        // Write the trace into a file.
        if(get_verbose() >= 0)
            operation::file_output_message<tim::project::rocprofsys>{}(
                get_file_name(), std::string{ "debug" });
    }
}

int64_t
get_tid()
{
    static thread_local auto _v = threading::get_id();
    return _v;
}
}  // namespace debug

template <typename Tp>
std::string
as_hex(Tp _v, size_t _width)
{
    std::stringstream _ss;
    _ss.fill('0');
    _ss << "0x" << std::hex << std::setw(_width) << _v;
    return _ss.str();
}

template <>
std::string
as_hex<address_range_t>(address_range_t _v, size_t _width)
{
    return (_v.is_range()) ? JOIN('-', as_hex(_v.low, _width), as_hex(_v.high, _width))
                           : as_hex(_v.low, _width);
}

template std::string as_hex<int32_t>(int32_t, size_t);
template std::string as_hex<uint32_t>(uint32_t, size_t);
template std::string as_hex<int64_t>(int64_t, size_t);
template std::string as_hex<uint64_t>(uint64_t, size_t);
template std::string
as_hex<void*>(void*, size_t);
}  // namespace rocprofsys
