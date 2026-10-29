# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Embed a JSON schema as a C++ string_view.
#
# cmake -DSCHEMA_INPUT=path/to/schema.json -DSCHEMA_OUTPUT=path/to/output.cpp
#       -P EmbedSchema.cmake

if(NOT DEFINED SCHEMA_INPUT)
    message(FATAL_ERROR "SCHEMA_INPUT not defined")
endif()

if(NOT DEFINED SCHEMA_OUTPUT)
    message(FATAL_ERROR "SCHEMA_OUTPUT not defined")
endif()

file(READ "${SCHEMA_INPUT}" SCHEMA_CONTENT)

file(
    WRITE "${SCHEMA_OUTPUT}"
    "// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Auto-generated from ${SCHEMA_INPUT}. Do not edit.

#include <string_view>

namespace rocprofsys::parser
{
std::string_view
get_embedded_schema_json()
{
    static constexpr std::string_view k_schema =
        R\"schema(${SCHEMA_CONTENT})schema\";
    return k_schema;
}
}  // namespace rocprofsys::parser
"
)

message(STATUS "Generated embedded schema: ${SCHEMA_OUTPUT}")
