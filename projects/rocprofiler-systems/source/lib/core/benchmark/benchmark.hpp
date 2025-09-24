#pragma once

#include "category.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys
{
namespace impl
{

template <typename, typename = void>
constexpr bool is_valid_type = false;

template <typename T>
constexpr bool is_valid_type<
    T, std::void_t<decltype(std::declval<T>().index), decltype(std::declval<T>().name)>> =
    true;

using tid_t = __pid_t;

struct indexed_category
{
    size_t category_index;
    tid_t  thread_id;

    friend bool operator==(const indexed_category& lhs, const indexed_category& rhs)
    {
        return lhs.category_index == rhs.category_index && lhs.thread_id == rhs.thread_id;
    }
};

struct indexed_category_hash
{
    size_t operator()(const indexed_category& p) const noexcept
    {
        std::size_t hash1 = std::hash<size_t>{}(p.category_index);
        std::size_t hash2 = std::hash<size_t>{}(p.thread_id);
        return hash1 ^ (hash2 << 1);
    }
};

struct result_data
{
    uint64_t total_time = 0;
    size_t   count      = 0;
    uint64_t min_time   = std::numeric_limits<uint64_t>::max();
    uint64_t max_time   = std::numeric_limits<uint64_t>::min();

    void update(uint64_t duration)
    {
        total_time += duration;
        count++;
        min_time = (duration < min_time) ? duration : min_time;
        max_time = (duration > max_time) ? duration : max_time;
    }
};

struct benchmark
{
private:
    using clock      = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;

public:
    template <typename... category>
    void start(category... start)
    {
        static_assert((is_valid_type<category> && ...),
                      "All categories must derive from ::benchmark::category::base");

        const auto      _now = clock::now();
        std::lock_guard _lock(m_mutex);

        (start_category(start, _now), ...);
    }

    template <typename... category>
    void end(category... end)
    {
        static_assert((is_valid_type<category> && ...),
                      "All categories must derive from ::benchmark::category::base");

        static const thread_local auto _thread_id = gettid();
        const auto                     _now       = clock::now();
        std::lock_guard                _lock(m_mutex);
        (end_category(end, _now), ...);
    }

    void show_results()
    {
        // std::sort(m_results.begin(), m_results.end(),
        //           [](const auto& a, const auto& b) { return a.total_time >
        //           b.total_time; });

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

        // Collect results with their index
        std::vector<std::pair<size_t, result_data>> sorted_results;
        for(size_t i = 0; i < m_results.size(); ++i)
        {
            if(m_results[i].count > 0)
            {
                sorted_results.emplace_back(i, m_results[i]);
            }
        }

        // Sort by total_time descending
        std::sort(sorted_results.begin(), sorted_results.end(),
                  [](const auto& a, const auto& b) {
                      return a.second.total_time > b.second.total_time;
                  });

        for(const auto& [index, result] : sorted_results)
        {
            double totalMs = static_cast<double>(result.total_time) / 1000.0;
            double avgUs   = result.count
                                 ? static_cast<double>(result.total_time) / result.count
                                 : 0.0;
            std::cout << std::left << std::setw(_category)
                      << rocprofsys::benchmark::category::to_string(index) << std::right
                      << std::setw(_calls) << result.count << std::setw(_total)
                      << std::fixed << std::setprecision(3) << totalMs << std::setw(_avg)
                      << std::fixed << std::setprecision(1) << avgUs << std::setw(_min)
                      << ((result.count != 0u) ? result.min_time : 0) << std::setw(_max)
                      << ((result.count != 0u) ? result.max_time : 0) << "\n";
        }

        std::cout << std::string(_category + _calls + _total + _avg + _min + _max, '=')
                  << "\033[0m"
                  << "\n\n";
    }

private:
    template <typename category>
    void start_category(category&& cat, const time_point& start_time)
    {
        static const thread_local auto _thread_id = gettid();
        if(m_started.find({ cat.index, _thread_id }) != m_started.end())
        {
            // ROCPROFSYS_WARNING(1, "Benchmark error: category already started!\n");
            return;
        }
        m_started[{ cat.index, _thread_id }] = start_time;
    };

    template <typename category>
    void end_category(category&& cat, const time_point& end_time)
    {
        static const thread_local auto _thread_id = gettid();

        auto _it = m_started.find({ cat.index, _thread_id });
        if(_it == m_started.end())
        {
            // ROCPROFSYS_WARNING(1, "Benchmark error: missing start time for
            // category!\n");
            return;
        }
        auto _category_index = _it->first.category_index;
        auto _time_point     = _it->second;
        auto duration        = std::chrono::duration_cast<std::chrono::microseconds>(
                            clock::now() - _time_point)
                            .count();
        m_results[_category_index].update(duration);
        m_started.erase(_it);
    }

private:
    std::unordered_map<indexed_category, time_point, indexed_category_hash> m_started;
    std::array<result_data, rocprofsys::benchmark::category::total_count>   m_results;
    std::mutex                                                              m_mutex;
};
}  // namespace impl

namespace benchmark
{
impl::benchmark&
get_benchmark();

template <typename... start_categories>
extern void
start(start_categories... start)
{
    get_benchmark().start(start...);
}

template <typename... start_categories>
extern void
end(start_categories... end)
{
    get_benchmark().end(end...);
}

void
show_results();

};  // namespace benchmark
}  // namespace rocprofsys
