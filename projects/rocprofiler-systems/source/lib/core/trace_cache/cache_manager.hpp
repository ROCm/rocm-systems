// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output_file_registry.hpp"
#include "core/progress/tracker.hpp"
#include "core/trace_cache/buffer_storage.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "core/trace_cache/storage_parser_alias.hpp"
#include "library/runtime.hpp"
#include <memory>
#include <unistd.h>

namespace rocprofsys
{
namespace trace_cache
{

using buffer_storage_t = buffer_storage<flush_worker_factory_t, type_identifier_t>;

class cache_manager
{
public:
    static cache_manager& get_instance();
    buffer_storage_t&     get_buffer_storage() { return m_storage; }
    metadata_registry&    get_metadata_registry() { return *m_metadata; }
    void                  shutdown();
    void                  post_process_bulk(output_file_registry& output_registry,
                                            progress::tracker&    tracker);

private:
    cache_manager() = default;

    buffer_storage_t m_storage{ utility::get_buffered_storage_filename(
        get_root_process_id(), getpid()) };

    std::shared_ptr<metadata_registry> m_metadata{
        std::make_shared<metadata_registry>()
    };
};

inline metadata_registry&
get_metadata_registry()
{
    return cache_manager::get_instance().get_metadata_registry();
}

inline buffer_storage_t&
get_buffer_storage()
{
    return cache_manager::get_instance().get_buffer_storage();
}

}  // namespace trace_cache
}  // namespace rocprofsys
