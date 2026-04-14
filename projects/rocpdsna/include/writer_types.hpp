// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

#include <rocpdsna/shared_types.hpp>

namespace rocpdsna::writer_types
{

/**
 * @brief OWNERSHIP MODEL
 *
 * All string fields in writer_types are NON-OWNING views (std::string_view).
 * The caller MUST ensure the underlying string data remains valid for the
 * duration of the register_*() or insert_*() call.
 *
 * SAFE patterns:
 *   - String literals: Always valid
 *   - std::string member variables: Valid while object exists
 *   - Function-local std::string: Valid within function scope
 *
 * UNSAFE patterns (will cause undefined behavior):
 *   - ss.str() temporaries: Destroyed immediately after statement
 *   - std::string(x).data(): Destroyed immediately after statement
 *
 * After register_*() returns, the writer owns internal copies and the
 * original string data may be freed.
 *
 * Fields use:
 *   - std::string_view: For NOT NULL fields (must have a value)
 *   - std::optional<std::string_view>: For nullable fields (can be NULL in DB)
 */

/**
 * @brief SCHEMA VERSION GUIDE
 *
 * This file defines structs used by both V3 and V4 schema writers.
 * The same struct is used for both versions -- the writer handles differences.
 *
 * HOW TO READ FIELD ANNOTATIONS:
 *   "V3/V4:"    : Used in both schema versions
 *   "V3 ONLY:"  : V3 only (ignored/absent in V4)
 *   "V4 ONLY:"  : V4 only (ignored/absent in V3)
 *   "V3:" / "V4:" on the same field : How each version interprets it differently
 *
 * IMPLICIT COLUMNS (not in data structs):
 *   The following SQL columns come from trace_environment_t, NOT from data structs:
 *     nid, pid, tid, agent_id, queue_id, stream_id  (V3: direct columns)
 *     track_id                                       (V4: replaces all of the above)
 *   These are passed as the second parameter to every insert_impl().
 *   See trace_environment_t for details.
 *
 * WRITER-DERIVED COLUMNS (auto-generated, not set by user):
 *   id (PK)            : C++ autoincrementer
 *   guid               : SQL DEFAULT
 *   start_id / end_id  : V4: writer calls insert_timestamp() internally
 *   name_id (V4 sample/alloc) : writer derives from track.name or data.name
 *
 * V4-ONLY TABLES (no V3 equivalent):
 *   rocpd_timestamp, rocpd_info_address_range, rocpd_info_source_code,
 *   rocpd_info_pc, rocpd_info_category, rocpd_line_info, rocpd_call_stack
 *
 * SHARED_TYPES (see shared_types.hpp):
 *   Types in shared_types.hpp are used ONLY in V4 for embedded data
 *   (call stacks, line info, source code). In V3, the same data is
 *   serialized to JSONB columns in rocpd_event.
 *
 * STRUCT FIELD GROUPING:
 *   Within each struct, fields are organized into sections:
 *     -- V3 & V4 fields --   Fields used in both schema versions
 *     -- V4-only fields --   Fields that only exist in V4 (ignored in V3)
 *     -- V3-only fields --   Fields that only exist in V3 (ignored in V4)
 *     -- FK context --       Fields supplied via trace_environment_t
 */

/***
 * @brief Node id
 * @note This is a unique value which will be used to identify the node
 */
using node_id_t = size_t;
/***
 * @brief Process id
 * @note This is a unique value which will be used to identify the process
 */
using process_id_t = size_t;
/***
 * @brief Thread id
 * @note This is a unique value which will be used to identify the thread
 */
using thread_id_t = size_t;

/***
 * @brief Code object id
 * @note This is a unique value which will be used to identify the code object
 */
using code_object_id_t = size_t;
/***
 * @brief Kernel symbol id
 * @note This is a unique value which will be used to identify the kernel symbol
 */
using kernel_symbol_id_t = size_t;
/***
 * @brief Pmc description name
 * @note This is a unique value which will be used to identify the pmc description
 */
using pmc_description_name_t = std::string_view;
/***
 * @brief Stream id
 * @note This is a unique value which will be used to identify the stream
 */
using stream_id_t = size_t;
/***
 * @brief Queue id
 * @note This is a unique value which will be used to identify the queue
 */
using queue_id_t = size_t;
/***
 * @brief Track name
 * @note This is a unique value which will be used to identify the track
 */
using track_name_t = std::string_view;

// =============================================================================
// v4+ ID Types
// =============================================================================

/*** @brief Category id - unique identifier for an event category  */
using category_id_t = size_t;

/*** @brief Address range id - unique identifier for an address range  */
using address_range_id_t = size_t;

/*** @brief Source code id - unique identifier for source code info  */
using source_code_id_t = size_t;

/*** @brief Program counter id - unique identifier for PC info  */
using pc_id_t = size_t;

/*** @brief Track id - unique identifier for a track record  */
using track_id_t = size_t;

/*** @brief Timestamp id - unique identifier for a timestamp record  */
using timestamp_id_t = size_t;

/*** @brief Event id - unique identifier for an event record  */
using event_id_t = size_t;

using timestamp_ns_t = size_t;

constexpr std::string_view empty_json = "{}";

/***
 * @brief Agent unique id
 * @note This is a struct which will be used to identify the agent uniquely.
 * @param logical_index Logical index which will uniquely identify the agent.
 * @param agent_type Agent type which will uniquely identify the agent.
 * @note Maps to rocpd_info_agent(anuj_done) columns: type, type_index (V3 & V4)
 */
struct agent_unique_id_t
{
    std::optional<std::string_view> agent_type;
    size_t                          type_index;

    bool operator==(const agent_unique_id_t& other) const noexcept
    {
        return agent_type == other.agent_type && type_index == other.type_index;
    }
};

/***
 * @brief Trace environment - execution context passed to every insert_impl().
 * @note NOT directly mapped to a single table. Provides the "where it happened"
 * context that data structs intentionally omit.
 *
 * SCHEMA COMPATIBILITY:
 *  Field      | V3 | V4 | Notes
 *  -----------+----+----+--------------------------------------
 *  node_id    |  Y |  Y | nid FK
 *  process_id |  Y |  Y | pid FK
 *  thread_id  |  Y |  Y | tid FK
 *  agent_id   |  Y |  Y | agent_id FK
 *  queue_id   |  Y |  Y | queue_id FK
 *  stream_id  |  Y |  Y | stream_id FK
 *  track_name |  Y |  Y | track name_id FK
 *  ppid       |  - |  Y | V4 only
 */
struct trace_environment_t
{
    std::optional<node_id_t>         node_id;
    std::optional<process_id_t>      process_id;
    std::optional<thread_id_t>       thread_id;
    std::optional<agent_unique_id_t> agent_id;
    std::optional<stream_id_t>       stream_id;
    std::optional<queue_id_t>        queue_id;
    std::optional<track_name_t>      track_name;
    std::optional<size_t>            ppid;
};

// --------------------- Info Tables ---------------------

/***
 * @brief Node info
 * @note Maps to rocpd_info_node table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field          | V3 | V4 | Notes
 *  ---------------+----+----+--------------------------------------
 *  node_id        |  Y |  Y | Internal key (NOT stored in DB)
 *  hash           |  Y |  Y | UNIQUE constraint
 *  machine_id     |  Y |  Y | UNIQUE constraint
 *  system_name    |  Y |  Y | Nullable
 *  hostname       |  Y |  Y | Nullable
 *  release        |  Y |  Y | Nullable
 *  version        |  Y |  Y | Nullable
 *  hardware_name  |  Y |  Y | Nullable
 *  domain_name    |  Y |  Y | Nullable
 *  name           |  - |  Y | V4 only
 */
struct node_info_t
{
    node_id_t                       node_id;
    size_t                          hash;
    std::string_view                machine_id;
    std::optional<std::string_view> system_name;
    std::optional<std::string_view> hostname;
    std::optional<std::string_view> release;
    std::optional<std::string_view> version;
    std::optional<std::string_view> hardware_name;
    std::optional<std::string_view> domain_name;
    std::optional<std::string_view> name;
};

/***
 * @brief Process info
 * @note Maps to rocpd_info_process table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field        | V3 | V4 | Notes
 *  -------------+----+----+--------------------------------------
 *  ppid         |  Y |  Y |
 *  pid          |  Y |  Y | NOT NULL
 *  init         |  Y |  Y |
 *  fini         |  Y |  Y |
 *  start        |  Y |  Y |
 *  end          |  Y |  Y |
 *  command      |  Y |  Y | Nullable
 *  environment  |  Y |  Y | Default "{}"
 *  extdata      |  Y |  Y | Default "{}"
 *  node_id      |  Y |  Y | nid FK
 *  name         |  - |  Y | V4 only
 */
struct process_info_t
{
    size_t                          ppid{};
    process_id_t                    pid{};
    size_t                          init{};
    size_t                          fini{};
    size_t                          start{};
    size_t                          end{};
    std::optional<std::string_view> command;
    std::string_view                environment = empty_json;
    std::string_view                extdata     = empty_json;
    node_id_t                       node_id{};
    std::optional<std::string_view> name;
};

/***
 * @brief Agent info
 * @note Maps to rocpd_info_agent table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field            | V3 | V4 | Notes
 *  -----------------+----+----+--------------------------------------
 *  unique_id        |  Y |  Y | type + type_index (identity key)
 *  absolute_index   |  Y |  Y |
 *  logical_index    |  Y |  Y |
 *  uuid             |  Y |  Y |
 *  name             |  Y |  Y |
 *  model_name       |  Y |  Y |
 *  vendor_name      |  Y |  Y |
 *  product_name     |  Y |  Y |
 *  extdata          |  Y |  Y | Default "{}"
 *  node_id          |  Y |  Y | nid FK
 *  process_id       |  Y |  Y | pid FK
 *  generic_name     |  - |  Y | V4 only
 *  user_name        |  Y |  - | V3 only
 */
struct agent_info_t
{
    agent_unique_id_t               unique_id{};
    size_t                          absolute_index{};
    size_t                          logical_index{};
    size_t                          uuid{};
    std::optional<std::string_view> name;
    std::optional<std::string_view> generic_name;
    std::optional<std::string_view> model_name;
    std::optional<std::string_view> vendor_name;
    std::optional<std::string_view> product_name;
    std::optional<std::string_view> user_name;
    std::string_view                extdata = empty_json;
    node_id_t                       node_id{};
    process_id_t                    process_id{};
};

/***
 * @brief Pmc info unique id
 * @note Used to uniquely identify a PMC counter by name and optional agent
 */
struct pmc_info_unique_id_t
{
    pmc_description_name_t           name;
    std::optional<agent_unique_id_t> agent_id;

    bool operator==(const pmc_info_unique_id_t& other) const noexcept
    {
        const bool are_names_same = name == other.name;
        if(agent_id.has_value() && other.agent_id.has_value())
        {
            return are_names_same && (agent_id.value() == other.agent_id.value());
        }
        return are_names_same;
    }
};

/***
 * @brief Pmc info
 * @note Maps to rocpd_info_pmc table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field             | V3 | V4 | Notes
 *  ------------------+----+----+--------------------------------------
 *  unique_id         |  Y |  Y | name + agent_id (identity key)
 *  target_arch       |  Y |  Y | CHECK IN ('CPU','GPU')
 *  event_code        |  Y |  Y |
 *  instance_id       |  Y |  Y |
 *  symbol            |  Y |  Y | NOT NULL
 *  description       |  Y |  Y | Nullable
 *  long_description  |  Y |  Y | Nullable
 *  component         |  Y |  Y | Nullable
 *  units             |  Y |  Y | Nullable
 *  value_type        |  Y |  Y | CHECK IN ('ABS','ACCUM','RELATIVE')
 *  block             |  Y |  Y | Nullable
 *  expression        |  Y |  Y | Nullable
 *  is_constant       |  Y |  Y |
 *  is_derived        |  Y |  Y |
 *  extdata           |  Y |  Y | Default "{}"
 *  node_id           |  Y |  Y | nid FK
 *  process_id        |  Y |  Y | pid FK
 *  qualifier         |  - |  Y | V4 only
 */
struct pmc_info_t
{
    pmc_info_unique_id_t            unique_id;
    std::optional<std::string_view> target_arch;
    size_t                          event_code{};
    size_t                          instance_id{};
    std::string_view                symbol;
    std::optional<std::string_view> description;
    std::optional<std::string_view> long_description;
    std::optional<std::string_view> component;
    std::optional<std::string_view> units;
    std::optional<std::string_view> value_type;
    std::optional<std::string_view> block;
    std::optional<std::string_view> expression;
    size_t                          is_constant{};
    size_t                          is_derived{};
    std::string_view                extdata = empty_json;
    node_id_t                       node_id{};
    process_id_t                    process_id{};
    std::optional<std::string_view> qualifier;
};

/***
 * @brief Thread info
 * @note Maps to rocpd_info_thread table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field              | V3 | V4 | Notes
 *  -------------------+----+----+--------------------------------------
 *  parent_process_id  |  Y |  Y | ppid (INTEGER)
 *  thread_id          |  Y |  Y | tid (NOT NULL)
 *  name               |  Y |  Y | Nullable
 *  start              |  Y |  Y | Nullable
 *  end                |  Y |  Y | Nullable
 *  extdata            |  Y |  Y | Default "{}"
 *  node_id            |  Y |  Y | nid FK
 *  process_id         |  Y |  Y | pid FK
 */
struct thread_info_t
{
    std::optional<size_t>           parent_process_id{};
    thread_id_t                     thread_id{};
    std::optional<std::string_view> name;
    std::optional<size_t>           start{};
    std::optional<size_t>           end{};
    std::string_view                extdata = empty_json;
    node_id_t                       node_id{};
    process_id_t                    process_id{};
};

/***
 * @brief Stream info
 * @note Maps to rocpd_info_stream table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field      | V3 | V4 | Notes
 *  -----------+----+----+--------------------------------------
 *  stream_id  |  Y |  Y | Internal key (NOT stored in DB)
 *  name       |  Y |  Y | Nullable
 *  extdata    |  Y |  Y | Default "{}"
 *  node_id    |  Y |  Y | nid FK
 *  process_id |  Y |  Y | pid FK
 */
struct stream_info_t
{
    stream_id_t                     stream_id{};
    std::optional<std::string_view> name;
    std::string_view                extdata = empty_json;
    node_id_t                       node_id{};
    process_id_t                    process_id{};
};

/***
 * @brief Queue info
 * @note Maps to rocpd_info_queue table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field      | V3 | V4 | Notes
 *  -----------+----+----+--------------------------------------
 *  queue_id   |  Y |  Y | Internal key (NOT stored in DB)
 *  name       |  Y |  Y | Nullable
 *  extdata    |  Y |  Y | Default "{}"
 *  node_id    |  Y |  Y | nid FK
 *  process_id |  Y |  Y | pid FK
 */
struct queue_info_t
{
    queue_id_t                      queue_id{};
    std::optional<std::string_view> name;
    std::string_view                extdata = empty_json;
    node_id_t                       node_id{};
    process_id_t                    process_id{};
};

/***
 * @brief Code object info
 * @note Maps to rocpd_info_code_object table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field         | V3 | V4 | Notes
 *  --------------+----+----+--------------------------------------
 *  id            |  Y |  Y | Internal key (NOT stored in DB)
 *  uri           |  Y |  Y | Nullable
 *  load_base     |  Y |  Y |
 *  load_size     |  Y |  Y |
 *  load_delta    |  Y |  Y |
 *  storage_type  |  Y |  Y | CHECK IN ('FILE','MEMORY')
 *  extdata       |  Y |  Y | Default "{}"
 *  node_id       |  Y |  Y | nid FK
 *  process_id    |  Y |  Y | pid FK
 *  agent_id      |  Y |  Y | FK (optional)
 */
struct code_object_info_t
{
    code_object_id_t                 id{};
    std::optional<std::string_view>  uri;
    size_t                           load_base{};
    size_t                           load_size{};
    size_t                           load_delta{};
    std::optional<std::string_view>  storage_type;
    std::string_view                 extdata = empty_json;
    node_id_t                        node_id{};
    process_id_t                     process_id{};
    std::optional<agent_unique_id_t> agent_id;
};

/***
 * @brief Kernel symbol info
 * @note Maps to rocpd_info_kernel_symbol table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field                      | V3 | V4 | Notes
 *  ---------------------------+----+----+--------------------------------------
 *  id                         |  Y |  Y | Internal key (NOT stored in DB)
 *  name                       |  Y |  Y | kernel_name column
 *  display_name               |  Y |  Y | Nullable
 *  kernel_object              |  Y |  Y |
 *  kernarg_segment_size       |  Y |  Y |
 *  kernarg_segment_alignment  |  Y |  Y |
 *  group_segment_size         |  Y |  Y |
 *  private_segment_size       |  Y |  Y |
 *  sgpr_count                 |  Y |  Y |
 *  arch_vgpr_count            |  Y |  Y |
 *  accum_vgpr_count           |  Y |  Y |
 *  extdata                    |  Y |  Y | Default "{}"
 *  node_id                    |  Y |  Y | nid FK
 *  process_id                 |  Y |  Y | pid FK
 *  code_obj_id                |  Y |  Y | code_object_id FK
 */
struct kernel_symbol_info_t
{
    kernel_symbol_id_t              id{};
    std::optional<std::string_view> name;
    std::optional<std::string_view> display_name;
    size_t                          kernel_object{};
    size_t                          kernarg_segment_size{};
    size_t                          kernarg_segment_alignment{};
    size_t                          group_segment_size{};
    size_t                          private_segment_size{};
    size_t                          sgpr_count{};
    size_t                          arch_vgpr_count{};
    size_t                          accum_vgpr_count{};
    std::string_view                extdata = empty_json;
    node_id_t                       node_id{};
    process_id_t                    process_id{};
    code_object_id_t                code_obj_id{};
};

// =============================================================================
// Info Tables - V4 Only (Category, Address Range, Source Code, PC)
// =============================================================================

/***
 * @brief Category info
 * @note Maps to rocpd_info_category table (V4 ONLY - no V3 equivalent)
 *
 * SCHEMA COMPATIBILITY:
 *  Field    | V3 | V4 | Notes
 *  ---------+----+----+--------------------------------------
 *  id       |  - |  Y | Internal key (NOT stored in DB)
 *  name     |  - |  Y | NOT NULL
 *  extdata  |  - |  Y | Default "{}"
 */
struct category_info_t
{
    category_id_t    id{};
    std::string_view name;
    std::string_view extdata = empty_json;
};

/***
 * @brief Address range info
 * @note Maps to rocpd_info_address_range table (V4 ONLY - no V3 equivalent)
 *
 * SCHEMA COMPATIBILITY:
 *  Field         | V3 | V4 | Notes
 *  --------------+----+----+--------------------------------------
 *  id            |  - |  Y | Internal key (NOT stored in DB)
 *  address_base  |  - |  Y |
 *  address_low   |  - |  Y | CHECK >= address_base
 *  address_high  |  - |  Y | CHECK >= address_low
 *  extdata       |  - |  Y | Default "{}"
 *  node_id       |  - |  Y | nid FK
 *  process_id    |  - |  Y | pid FK
 */
struct address_range_info_t
{
    address_range_id_t id{};
    size_t             address_base{};
    size_t             address_low{};
    size_t             address_high{};
    std::string_view   extdata = empty_json;
    node_id_t          node_id{};
    process_id_t       process_id{};
};

/***
 * @brief Source code info
 * @note Maps to rocpd_info_source_code table (V4 ONLY - no V3 equivalent)
 *
 * SCHEMA COMPATIBILITY:
 *  Field         | V3 | V4 | Notes
 *  --------------+----+----+--------------------------------------
 *  id            |  - |  Y | Internal key (NOT stored in DB)
 *  file          |  - |  Y | Nullable
 *  line_number   |  - |  Y | Nullable
 *  lines         |  - |  Y | JSONB DEFAULT "[]"
 *  instructions  |  - |  Y | JSONB DEFAULT "[]"
 *  extdata       |  - |  Y | Default "{}"
 *  node_id       |  - |  Y | nid FK
 *  process_id    |  - |  Y | pid FK
 *  address_id    |  - |  Y | FK to rocpd_info_address_range
 */
struct source_code_info_t
{
    source_code_id_t                  id{};
    std::optional<std::string_view>   file;
    std::optional<size_t>             line_number;
    std::string_view                  lines        = empty_json;
    std::string_view                  instructions = empty_json;
    std::string_view                  extdata      = empty_json;
    node_id_t                         node_id{};
    process_id_t                      process_id{};
    std::optional<address_range_id_t> address_id;
};

/***
 * @brief Program counter info
 * @note Maps to rocpd_info_pc table (V4 ONLY - no V3 equivalent)
 *
 * SCHEMA COMPATIBILITY:
 *  Field       | V3 | V4 | Notes
 *  ------------+----+----+--------------------------------------
 *  id          |  - |  Y | Internal key (NOT stored in DB)
 *  function    |  - |  Y | NOT NULL
 *  file        |  - |  Y | Nullable
 *  line        |  - |  Y | Nullable
 *  extdata     |  - |  Y | Default "{}"
 *  node_id     |  - |  Y | nid FK
 *  process_id  |  - |  Y | pid FK
 *  address_id  |  - |  Y | FK to rocpd_info_address_range
 */
struct pc_info_t
{
    pc_id_t                           id{};
    std::string_view                  function;
    std::optional<std::string_view>   file;
    std::optional<size_t>             line;
    std::string_view                  extdata = empty_json;
    node_id_t                         node_id{};
    process_id_t                      process_id{};
    std::optional<address_range_id_t> address_id;
};

/***
 * @brief Track info
 * @note Maps to rocpd_track table (V3 & V4, with V4 having additional columns)
 *
 * SCHEMA COMPATIBILITY:
 *  Field       | V3 | V4 | Notes
 *  ------------+----+----+--------------------------------------
 *  name        |  Y |  Y | name_id FK to rocpd_string
 *  extdata     |  Y |  Y | Default "{}"
 *  node_id     |  Y |  Y | nid FK
 *  process_id  |  Y |  Y | pid FK (optional)
 *  thread_id   |  Y |  Y | tid FK (optional)
 *  ppid        |  - |  Y | V4 only (INTEGER)
 *  agent_id    |  - |  Y | V4 only FK
 *  queue_id    |  - |  Y | V4 only FK
 *  stream_id   |  - |  Y | V4 only FK
 */
struct track_info_t
{
    std::optional<track_name_t> name;
    std::string_view            extdata = empty_json;

    node_id_t                   node_id{};
    std::optional<process_id_t> process_id;
    std::optional<thread_id_t>  thread_id;

    std::optional<size_t>            ppid;
    std::optional<agent_unique_id_t> agent_id;
    std::optional<queue_id_t>        queue_id;
    std::optional<stream_id_t>       stream_id;

    bool operator==(const track_info_t& other) const noexcept
    {
        return name == other.name && node_id == other.node_id &&
               process_id == other.process_id && thread_id == other.thread_id &&
               ppid == other.ppid && agent_id == other.agent_id &&
               queue_id == other.queue_id && stream_id == other.stream_id;
    }
};

// --------------------- Data Tables ---------------------

/***
 * @brief Function argument data for API tracing.
 * @note Maps to rocpd_arg table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field     | V3 | V4 | Notes
 *  ----------+----+----+--------------------------------------
 *  position  |  Y |  Y | NOT NULL
 *  type      |  Y |  Y | NOT NULL
 *  name      |  Y |  Y | NOT NULL
 *  value     |  Y |  Y | Nullable
 *  extdata   |  Y |  Y | Default "{}"
 *  (event_id)|  Y |  Y | FK set by writer, not in struct
 */
struct arg_data_t
{
    size_t                          position{};
    std::string_view                type;
    std::string_view                name;
    std::optional<std::string_view> value;
    std::string_view                extdata = empty_json;
};

/***
 * @brief Common event metadata shared by all profiling events.
 * @note Maps to rocpd_event table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field            | V3 | V4 | Notes
 *  -----------------+----+----+--------------------------------------
 *  stack_id         |  Y |  Y | Nullable
 *  parent_stack_id  |  Y |  Y | Nullable
 *  correlation_id   |  Y |  Y | Nullable
 *  call_stack       |  Y |  Y | V3: JSONB column; V4: rocpd_call_stack rows
 *  line_info_list   |  Y |  Y | V3: JSONB column; V4: rocpd_line_info rows
 *  event_category   |  Y |  Y | V3: FK to rocpd_string; V4: FK to rocpd_info_category
 *  extdata          |  Y |  Y | Default "{}"
 *  (nid, pid)       |  - |  Y | V4 only; supplied via trace_environment_t
 */
struct event_data_t
{
    std::optional<size_t> stack_id;
    std::optional<size_t> parent_stack_id;
    std::optional<size_t> correlation_id;

    shared_types::call_stack_t          call_stack;
    shared_types::source_context_list_t line_info_list;

    std::optional<std::string_view> event_category;
    std::string_view                extdata = empty_json;
};

/***
 * @brief A named time region representing a span of execution.
 * @note Maps to rocpd_region table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field            | V3 | V4 | Notes
 *  -----------------+----+----+--------------------------------------
 *  event            |  Y |  Y | Optional; becomes event_id FK
 *  start_timestamp  |  Y |  Y | V3: direct BIGINT; V4: timestamp FK
 *  end_timestamp    |  Y |  Y | V3: direct BIGINT; V4: timestamp FK
 *  name             |  Y |  Y | name_id FK to rocpd_string
 *  extdata          |  Y |  Y | Default "{}"
 *  args             |  Y |  Y | rocpd_arg rows (FK via event_id)
 *  (trace_env)      |  Y |  Y | V3: nid/pid/tid cols; V4: track_id FK
 */
struct region_data_t
{
    std::optional<event_data_t> event;
    timestamp_ns_t              start_timestamp;
    timestamp_ns_t              end_timestamp;
    std::string_view            name;
    std::string_view            extdata = empty_json;
    std::vector<arg_data_t>     args;
};

/***
 * @brief A point-in-time sample (instantaneous event).
 * @note Maps to rocpd_sample table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field        | V3 | V4 | Notes
 *  -------------+----+----+--------------------------------------
 *  timestamp    |  Y |  Y | V3: direct BIGINT; V4: timestamp FK
 *  track        |  Y |  Y | track_id FK
 *  extdata      |  Y |  Y | Default "{}"
 *  (name_id)    |  - |  Y | V4 only; derived from track.name
 *  (event_id)   |  Y |  Y | FK set by writer, not in struct
 */
struct sample_data_t
{
    timestamp_ns_t   timestamp{};
    track_info_t     track;
    std::string_view extdata = empty_json;
};

/***
 * @brief Performance counter (PMC) event data.
 * @note Maps to rocpd_pmc_event table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field     | V3 | V4 | Notes
 *  ----------+----+----+--------------------------------------
 *  event     |  Y |  Y | Optional; becomes event_id FK
 *  value     |  Y |  Y | REAL DEFAULT 0.0
 *  extdata   |  Y |  Y | Default "{}"
 *  sample    |  Y |  Y | Provides track_id and timestamp
 *  (pmc_id)  |  Y |  Y | FK set by writer from pmc_info_unique_id_t
 */
struct pmc_event_data_t
{
    std::optional<event_data_t> event;
    double                      value{};
    std::string_view            extdata = empty_json;
    sample_data_t               sample;
};

/***
 * @brief GPU kernel dispatch event data.
 * @note Maps to rocpd_kernel_dispatch table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field                | V3 | V4 | Notes
 *  ---------------------+----+----+--------------------------------------
 *  event                |  Y |  Y | Optional; becomes event_id FK
 *  dispatch_id          |  Y |  Y | NOT NULL
 *  start_timestamp      |  Y |  Y | V3: direct BIGINT; V4: timestamp FK
 *  end_timestamp        |  Y |  Y | V3: direct BIGINT; V4: timestamp FK
 *  kernel_symbol_id     |  Y |  Y | kernel_id FK
 *  code_object_id       |  Y |  Y | NOT stored (implied by kernel_symbol)
 *  private_segment_size |  Y |  Y |
 *  group_segment_size   |  Y |  Y |
 *  workgroup_size_x/y/z |  Y |  Y | NOT NULL
 *  grid_size_x/y/z      |  Y |  Y | NOT NULL
 *  name                 |  Y |  Y | region_name_id FK to rocpd_string
 *  extdata              |  Y |  Y | Default "{}"
 *  (trace_env)          |  Y |  Y | V3: nid/pid/tid/agent/queue/stream cols
 *                       |    |    | V4: track_id FK
 */
struct kernel_dispatch_data_t
{
    std::optional<event_data_t>     event;
    size_t                          dispatch_id{};
    timestamp_ns_t                  start_timestamp{};
    timestamp_ns_t                  end_timestamp{};
    kernel_symbol_id_t              kernel_symbol_id{};
    code_object_id_t                code_object_id{};
    size_t                          private_segment_size{};
    size_t                          group_segment_size{};
    size_t                          workgroup_size_x{};
    size_t                          workgroup_size_y{};
    size_t                          workgroup_size_z{};
    size_t                          grid_size_x{};
    size_t                          grid_size_y{};
    size_t                          grid_size_z{};
    std::optional<std::string_view> name;
    std::string_view                extdata = empty_json;
};

/***
 * @brief Memory copy operation event data.
 * @note Maps to rocpd_memory_copy table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field            | V3 | V4 | Notes
 *  -----------------+----+----+--------------------------------------
 *  event            |  Y |  Y | Optional; becomes event_id FK
 *  start_timestamp  |  Y |  Y | V3: direct BIGINT; V4: timestamp FK
 *  end_timestamp    |  Y |  Y | V3: direct BIGINT; V4: timestamp FK
 *  dst_agent_id     |  Y |  Y | FK (optional)
 *  dst_address      |  Y |  Y | Nullable
 *  src_agent_id     |  Y |  Y | FK (optional)
 *  src_address      |  Y |  Y | Nullable
 *  size             |  Y |  Y | NOT NULL
 *  name             |  Y |  Y | name_id FK to rocpd_string
 *  region_name      |  Y |  Y | region_name_id FK (optional)
 *  extdata          |  Y |  Y | Default "{}"
 *  (trace_env)      |  Y |  Y | V3: nid/pid/tid/queue/stream cols
 *                   |    |    | V4: track_id FK
 */
struct memory_copy_data_t
{
    std::optional<event_data_t>      event;
    timestamp_ns_t                   start_timestamp{};
    timestamp_ns_t                   end_timestamp{};
    std::optional<agent_unique_id_t> dst_agent_id;
    std::optional<size_t>            dst_address;
    std::optional<agent_unique_id_t> src_agent_id;
    std::optional<size_t>            src_address;
    size_t                           size{};
    std::string_view                 name;
    std::optional<std::string_view>  region_name;
    std::string_view                 extdata = empty_json;
};

/***
 * @brief Memory allocation event data.
 * @note Maps to rocpd_memory_allocate table (V3 & V4)
 *
 * SCHEMA COMPATIBILITY:
 *  Field            | V3 | V4 | Notes
 *  -----------------+----+----+--------------------------------------
 *  event            |  Y |  Y | Optional; becomes event_id FK
 *  type             |  Y |  Y | CHECK IN ('ALLOC','FREE','REALLOC','RECLAIM')
 *  level            |  Y |  Y | CHECK IN ('REAL','VIRTUAL','SCRATCH')
 *  start_timestamp  |  Y |  Y | V3: direct BIGINT; V4: timestamp FK
 *  end_timestamp    |  Y |  Y | V3: direct BIGINT; V4: timestamp FK
 *  address          |  Y |  Y | Nullable
 *  size             |  Y |  Y | NOT NULL
 *  extdata          |  Y |  Y | Default "{}"
 *  name             |  - |  Y | V4 only; fallback: type or generic
 *  region_name      |  - |  Y | V4 only; nullable
 *  (trace_env)      |  Y |  Y | V3: nid/pid/tid/agent/queue/stream cols
 *                   |    |    | V4: track_id FK
 */
struct memory_alloc_data_t
{
    std::optional<event_data_t>     event;
    std::optional<std::string_view> type;
    std::optional<std::string_view> level;
    timestamp_ns_t                  start_timestamp{};
    timestamp_ns_t                  end_timestamp{};
    std::optional<size_t>           address;
    size_t                          size{};
    std::string_view                extdata = empty_json;

    std::optional<std::string_view> name;
    std::optional<std::string_view> region_name;
};

}  // namespace rocpdsna::writer_types
