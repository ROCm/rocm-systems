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

#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "core/benchmark/category.hpp"
#include "core/debug.hpp"

namespace rocprofsys
{
namespace benchmark
{
namespace
{
template <bool enabled, typename category_enum, category_enum... enabled_categories>
struct benchmark_impl
{
    template <category_enum... categories>
    struct scope
    {
        scope(const scope&)            = delete;
        scope& operator=(const scope&) = delete;
        ~scope()                       = default;

    protected:
        scope()                   = default;
        scope(scope&&)            = default;
        scope& operator=(scope&&) = default;
    };

    template <category_enum... categories>
    static void start()
    {}

    template <category_enum... categories>
    static void end()
    {}

    template <category_enum... categories>
    [[nodiscard]] static scope<categories...> scoped_trace()
    {
        return scope<categories...>{};
    }

    static void init_from_env(const char* = nullptr) {}
    static void show_results() {}
};

using tid_t = __pid_t;
struct indexed_category
{
    size_t category;
    tid_t  thread_id;

    friend bool operator==(const indexed_category& lhs, const indexed_category& rhs)
    {
        return lhs.category == rhs.category && lhs.thread_id == rhs.thread_id;
    }
};

struct indexed_category_hash
{
    size_t operator()(const indexed_category& p) const noexcept
    {
        std::size_t hash1 = std::hash<size_t>{}(p.category);
        std::size_t hash2 = std::hash<size_t>{}(p.thread_id);
        return hash1 ^ (hash2 << 1);
    }
};

template <typename category_enum, category_enum... enabled_categories>
struct benchmark_impl<true, category_enum, enabled_categories...>
{
    static_assert(std::is_enum_v<category_enum>, "category_enum must be an enum");

public:
    using clock                             = std::chrono::high_resolution_clock;
    using time_point                        = clock::time_point;
    static constexpr size_t _max_categories = static_cast<size_t>(category_enum::count);

    template <category_enum... categories>
    struct scope
    {
        friend benchmark_impl;

    public:
        scope(const scope&)            = delete;
        scope& operator=(const scope&) = delete;
        ~scope() { end<categories...>(); }

    protected:
        scope() { start<categories...>(); }

        scope(scope&&)            = default;
        scope& operator=(scope&&) = default;
    };

    template <category_enum... categories>
    static void start()
    {
        static const thread_local auto _thread_id = gettid();
        const auto                     now        = clock::now();
        std::lock_guard                lock(m_mutex);
        (..., (is_category_defined<categories>([&] {
             if(m_enabled.test(to_index(categories)))
                 m_started[{ to_index(categories), _thread_id }] = now;
         })));
    }

    template <category_enum... categories>
    static void end()
    {
        static const thread_local auto _thread_id = getpid();
        const auto                     _end_time  = clock::now();
        std::lock_guard                lock(m_mutex);
        (..., (is_category_defined<categories>([&] {
             if(m_enabled.test(to_index(categories)))
                 end_category(_end_time, categories, _thread_id);
         })));
    }

    template <category_enum... categories>
    [[nodiscard]] static scope<categories...> scoped_trace()
    {
        return scope<categories...>{};
    }

    static void init_from_env(const char* envVar = "ROCPROFSYS_BENCHMARK_CATEGORIES")
    {
        std::lock_guard lock(m_mutex);
        const auto*     env = std::getenv(envVar);
        if(env == nullptr || std::string(env).empty())
        {
            ROCPROFSYS_WARNING(1, "No BENCHMARK categories specified in environment "
                                  "variable ROCPROFSYS_BENCHMARK_CATEGORIES.\n");
            return;
        }
        std::string        _str(env);
        std::istringstream ss(_str);
        std::string        token;

        while(std::getline(ss, token, ','))
        {
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
            for(category_enum cat : compiledCategories)
            {
                if(to_string(cat) == token)
                {
                    m_enabled.set(to_index(cat));
                }
            }
        }
    }

    static void show_results()
    {
        std::lock_guard                                    lock(m_mutex);
        std::vector<std::pair<category_enum, result_data>> sorted;

        for(category_enum cat : compiledCategories)
        {
            const auto& data = m_results[to_index(cat)];
            if(data.count > 0)
            {
                sorted.emplace_back(cat, data);
            }
        }

        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.second.total_time > b.second.total_time;
        });

        constexpr uint32_t _category = 30;
        constexpr uint32_t _calls    = 8;
        constexpr uint32_t _total    = 12;
        constexpr uint32_t _avg      = 10;
        constexpr uint32_t _min      = 10;
        constexpr uint32_t _max      = 10;

        std::cout << "\033[32m"
                  << std::string(_category + _calls + _total + _avg + _min + _max, '=')
                  << "\n";
        std::cout << "Benchmark Results (Sorted by Total Time):\n";
        std::cout << std::string(_category + _calls + _total + _avg + _min + _max, '-')
                  << "\n";
        std::cout << std::left << std::setw(_category) << "Category" << std::right
                  << std::setw(_calls) << "Calls" << std::setw(_total) << "Total(ms)"
                  << std::setw(_avg) << "Avg(us)" << std::setw(_min) << "Min(us)"
                  << std::setw(_max) << "Max(us)" << "\n";

        std::cout << std::string(_category + _calls + _total + _avg + _min + _max, '-')
                  << "\n";

        for(const auto& [cat, data] : sorted)
        {
            double totalMs = static_cast<double>(data.total_time) / 1000.0;
            double avgUs   = static_cast<double>(data.total_time) / data.count;

            std::cout << std::left << std::setw(_category) << to_string(cat) << std::right
                      << std::setw(_calls) << data.count << std::setw(_total)
                      << std::fixed << std::setprecision(3) << totalMs << std::setw(_avg)
                      << std::fixed << std::setprecision(1) << avgUs << std::setw(_min)
                      << data.min_time << std::setw(_max) << data.max_time << "\n";
        }

        std::cout << std::string(_category + _calls + _total + _avg + _min + _max, '=')
                  << "\033[0m" << "\n\n";
    }

private:
    struct result_data
    {
        uint64_t total_time = 0;
        size_t   count      = 0;
        uint64_t min_time   = std::numeric_limits<uint64_t>::max();
        uint64_t max_time   = std::numeric_limits<uint64_t>::min();

        void update(uint64_t duration)
        {
            total_time += duration;
            count += 1;
            if(duration < min_time) min_time = duration;
            if(duration > max_time) max_time = duration;
        }
    };

    static constexpr size_t to_index(category_enum cat)
    {
        return static_cast<size_t>(cat);
    }

    static void end_category(const time_point& end_time, category_enum cat,
                             const tid_t thread_id)
    {
        const size_t _idx = to_index(cat);
        auto         _it  = m_started.find({ _idx, thread_id });
        if(_it == m_started.end())
        {
            ROCPROFSYS_WARNING(1, "Benchmark error: missing start time for category!\n");
            return;
        }

        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - _it->second)
                .count();
        m_started.erase(_it);
        m_results[_idx].update(duration);
    }

    template <category_enum Cat, typename Func>
    static constexpr void is_category_defined(Func&& f)
    {
        if constexpr(((Cat == enabled_categories) || ...))
        {
            f();
        }
    }

    static constexpr std::array<category_enum, sizeof...(enabled_categories)>
        compiledCategories = { enabled_categories... };

    static inline std::unordered_map<indexed_category, time_point, indexed_category_hash>
                                                           m_started;
    static inline std::array<result_data, _max_categories> m_results{};
    static inline std::bitset<_max_categories>             m_enabled;
    static inline std::mutex                               m_mutex;
};

#ifdef ROCPROFSYS_ENABLE_BENCHMARK
using _benchmark_impl = benchmark::benchmark_impl<
    static_cast<bool>(ROCPROFSYS_ENABLE_BENCHMARK), benchmark::category,
    benchmark::category::kernel_dispatch, benchmark::category::memory_copy,
    benchmark::category::memory_allocate, benchmark::category::db_entry_kernel_dispatch,
    benchmark::category::db_entry_memory_copy,
    benchmark::category::db_entry_memory_allocate,
    benchmark::category::perfetto_kernel_dispatch,
    benchmark::category::sdk_tool_buffered_tracing>;
#else
using _benchmark_impl = benchmark::benchmark_impl<false, benchmark::category>;
#endif
}  // namespace

template <category... categories>
void
start()
{
    _benchmark_impl::template start<categories...>();
}

template <category... categories>
void
end()
{
    _benchmark_impl::template end<categories...>();
}

template <category... categories>
[[nodiscard]] auto
scoped_trace()
{
    return _benchmark_impl::template scoped_trace<categories...>();
}

inline void
init_from_env(const char* envVar = "BENCHMARK_CATEGORIES")
{
    _benchmark_impl::init_from_env(envVar);
}

inline void
show_results()
{
    _benchmark_impl::show_results();
}

}  // namespace benchmark
}  // namespace rocprofsys
