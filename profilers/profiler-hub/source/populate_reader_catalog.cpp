// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "populate_reader_catalog.hpp"

#include "common/connection.hpp"
#include "common/connection_pool.hpp"
#include "common/thread_pool.hpp"
#include "profiler-hub/cpp/reader.hpp"
#include "reader_catalog.hpp"

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
run_all(common::thread_pool& workers, const std::vector<std::function<void()>>& jobs)
{
    std::mutex         error_mutex;
    std::exception_ptr first_error;

    std::vector<common::thread_pool::task_handle> handles;
    handles.reserve(jobs.size());

    for(const auto& job : jobs)
    {
        handles.push_back(
            workers.submit([&job, &error_mutex, &first_error](const std::stop_token&) {
                try
                {
                    job();
                } catch(...)
                {
                    std::scoped_lock lock{ error_mutex };
                    if(!first_error)
                    {
                        first_error = std::current_exception();
                    }
                }
            }));
    }

    for(const auto& handle : handles)
    {
        handle.wait();
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

    const auto job = [&](std::function<void(common::connection&)> step) {
        return std::function<void()>{ [&connections, step = std::move(step)] {
            connections.run_sync(step);
        } };
    };

    const auto chain = [&](std::vector<category_t> categories) {
        return job(
            [&catalog, categories = std::move(categories)](common::connection& conn) {
                for(const auto category : categories)
                {
                    conn.reader().build_catalog_category(category, catalog);
                }
            });
    };

    run_all(workers, { job([](common::connection& conn) {
                conn.reader().ensure_track_topology_indexes();
            }) });

    run_all(workers,
            {
                chain({
                    category_t::string_list,
                    category_t::nodes,
                    category_t::processes,
                    category_t::threads,
                    category_t::agents,
                }),
            });

    run_all(workers,
            {
                chain({ category_t::tracks }),
                chain({ category_t::code_objects, category_t::kernel_symbols }),
                chain({ category_t::streams }),
                chain({ category_t::queues }),
                chain({ category_t::pmc_infos }),
            });
}

}  // namespace profiler_hub
