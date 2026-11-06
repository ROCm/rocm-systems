// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/version.hpp"
#include "profiler-hub/writer.hpp"

#include "../errors/validation_error.hpp"
#include "../pmc_event_input.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace profiler_hub::schema
{

// One column the caller set. enum_value is compared to a CHECK (... IN ...) list
// when the parsed column has one.
struct field_assignment
{
    std::string                name;
    bool                       set = false;
    std::optional<std::string> enum_value;
};

namespace schema_validator
{

[[nodiscard]] validation_result
validate(const pmc_event_input_t& input, profiler_hub::version_t version);

[[nodiscard]] validation_result
validate_fields(std::string_view                     object_name,
                std::string_view                     table_name,
                profiler_hub::version_t              version,
                const std::vector<field_assignment>& fields);

// Validates first. insert_pmc_event_data runs only when validation succeeds.
[[nodiscard]] validation_result
write_pmc_event(profiler_hub::writer_t&                                 writer,
                const pmc_event_input_t&                                input,
                profiler_hub::version_t                                 version,
                const profiler_hub::writer_types::pmc_event_data_t&     event,
                const profiler_hub::writer_types::pmc_info_unique_id_t& pmc_unique_id);

}  // namespace schema_validator

}  // namespace profiler_hub::schema
