// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "writers/schema_v4/common_insert_operations.hpp"
#include "writers/schema_v4/info_registration_writer.hpp"
#include "writers/schema_v4/kernel_dispatch_writer.hpp"
#include "writers/schema_v4/memory_alloc_writer.hpp"
#include "writers/schema_v4/memory_copy_writer.hpp"
#include "writers/schema_v4/pmc_event_writer.hpp"
#include "writers/schema_v4/region_writer.hpp"

#include "data_storage/schema_v4/insert_statements.hpp"
#include "data_storage/schema_version.hpp"

namespace rocpdsna
{

/**
 * @brief Writer policy for schema v4 (latest)
 *
 * Key differences from v3:
 * - Uses track_id instead of nid/pid/tid columns
 * - Uses timestamp tables (rocpd_timestamp) instead of direct timestamps
 * - Event has no call_stack/line_info columns (stored in separate tables)
 * - Agent info uses generic_name instead of user_name
 */
struct writer_policy_v4
{
    using schema_tag_t = data_storage::schema_v4_tag;
    using insert_statements_t =
        data_storage::schema_v4::insert_statements<data_storage::sqlite_backend>;
    using common_ops_t             = common_insert_operations<schema_tag_t>;
    using info_writer_t            = info_registration_writer<schema_tag_t>;
    using region_writer_t          = region_writer<schema_tag_t>;
    using kernel_dispatch_writer_t = kernel_dispatch_writer<schema_tag_t>;
    using memory_copy_writer_t     = memory_copy_writer<schema_tag_t>;
    using memory_alloc_writer_t    = memory_alloc_writer<schema_tag_t>;
    using pmc_event_writer_t       = pmc_event_writer<schema_tag_t>;
};

}  // namespace rocpdsna
