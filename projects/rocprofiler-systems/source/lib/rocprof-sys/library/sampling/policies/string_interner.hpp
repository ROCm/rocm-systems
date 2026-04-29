// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>
#include <set>
#include <string>

namespace rocprofsys::sampling
{

// Thread-safe string interner producing stable `const char*` for the lifetime
// of the interner. Used by perfetto sinks where deferred track-event writes
// require pointer-stable strings.
class string_interner
{
public:
    [[nodiscard]] char const* intern(std::string const& value)
    {
        std::lock_guard<std::mutex> const lock{ m_mutex };
        return m_pool.insert(value).first->c_str();
    }

private:
    std::mutex            m_mutex;
    std::set<std::string> m_pool;
};

}  // namespace rocprofsys::sampling
