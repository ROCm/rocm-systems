// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "populate_reader_catalog.hpp"

#include "common/connection.hpp"
#include "common/connection_pool.hpp"
#include "common/helper_group.hpp"
#include "common/thread_pool.hpp"
#include "profiler-hub/cpp/reader.hpp"
#include "reader_catalog.hpp"

#include <atomic>
#include <exception>
#include <functional>
#include <mutex>
#include <stop_token>
#include <utility>
#include <vector>

namespace profiler_hub
{

namespace
{

void
run_all_and_rethrow(common::thread_pool&                      workers,
                    const std::vector<std::function<void()>>& jobs)
{
    std::atomic<size_t> next{ 0 };
    std::mutex          error_mutex;
    std::exception_ptr  first_error;

    const auto run_jobs = [&] {
        for(size_t i = next.fetch_add(1); i < jobs.size(); i = next.fetch_add(1))
        {
            try
            {
                jobs[i]();
            } catch(...)
            {
                const std::scoped_lock lock{ error_mutex };
                if(!first_error) first_error = std::current_exception();
            }
        }
    };

    {
        common::helper_group helpers{ workers };
        for(size_t i = 1; i < jobs.size(); ++i)
        {
            helpers.add([&run_jobs](const std::stop_token&) { run_jobs(); });
        }
        run_jobs();
    }

    if(first_error)
    {
        std::rethrow_exception(first_error);
    }
}

}  // namespace

void
populate_reader_catalog(common::thread_pool&     workers,
                        common::connection_pool& connections,
                        reader_catalog_t&        catalog)
{
    using category_t = reader_t::catalog_category_t;

    const auto make_step_task = [&](std::function<void(common::connection&)> step) {
        return std::function<void()>{ [&connections, step = std::move(step)] {
            connections.with_connection(step);
        } };
    };

    const auto make_sequential_task = [&](std::vector<category_t> categories) {
        return make_step_task(
            [&catalog, categories = std::move(categories)](common::connection& conn) {
                for(const auto category : categories)
                {
                    conn.reader().build_catalog_category(category, catalog);
                }
            });
    };

    run_all_and_rethrow(workers, { make_step_task([](common::connection& conn) {
                            conn.reader().ensure_track_topology_indexes();
                        }) });

    run_all_and_rethrow(workers,
                        {
                            make_sequential_task({
                                category_t::string_list,
                                category_t::nodes,
                                category_t::processes,
                                category_t::threads,
                                category_t::agents,
                            }),
                        });

    run_all_and_rethrow(workers,
                        {
                            make_sequential_task({ category_t::tracks }),
                            make_sequential_task(
                                { category_t::code_objects, category_t::kernel_symbols }),
                            make_sequential_task({ category_t::streams }),
                            make_sequential_task({ category_t::queues }),
                            make_sequential_task({ category_t::pmc_infos }),
                        });
}

}  // namespace profiler_hub
