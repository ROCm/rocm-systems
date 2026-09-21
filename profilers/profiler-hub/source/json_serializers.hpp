// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/reader_types.hpp"
#include "profiler-hub/shared_types.hpp"

#include <nlohmann/json_fwd.hpp>

#include <string>

namespace profiler_hub::json_serializers
{

using json_t = nlohmann::json;

/**
 * @brief Serialize address_range_info_t to JSON
 */
json_t
to_json(const profiler_hub::shared_types::address_range_info_t& addr_range);

/**
 * @brief Serialize program_counter_info_t to JSON
 */
json_t
to_json(const profiler_hub::shared_types::program_counter_info_t& pc_info);

/**
 * @brief Serialize stack_frame_t to JSON
 */
json_t
to_json(const profiler_hub::shared_types::stack_frame_t& frame);

/**
 * @brief Serialize call_stack_t to JSON string
 * @param call_stack The call stack to serialize (deque of stack frames)
 * @return JSON string representation of the call stack
 */
std::string
serialize_call_stack(const profiler_hub::shared_types::call_stack_t& call_stack);

/**
 * @brief Serialize source_code_info_t to JSON
 */
json_t
to_json(const profiler_hub::shared_types::source_code_info_t& source_code);

/**
 * @brief Serialize line_info_entry_t to JSON
 */
json_t
to_json(const profiler_hub::shared_types::line_info_entry_t& line_info);

/**
 * @brief Serialize source_context_list_t to JSON string
 * @param line_info_list The source context list to serialize
 * @return JSON string representation of the source context
 */
std::string
serialize_source_context(
    const profiler_hub::shared_types::source_context_list_t& line_info_list);

/**
 * @brief Deserialize call_stack_t from JSON string
 * @param json JSON string containing call stack data
 * @return Parsed call stack (empty if json is empty or malformed)
 */
profiler_hub::reader_types::call_stack_t
deserialize_call_stack(const std::string& json);

/**
 * @brief Deserialize source_context_list_t from JSON string
 * @param json JSON string containing source context data
 * @return Parsed source context list (empty if json is empty or malformed)
 */
profiler_hub::reader_types::source_context_list_t
deserialize_source_context(const std::string& json);

}  // namespace profiler_hub::json_serializers
