# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Compile FlatBuffers schemas into C++ headers and embed the simulation config
# schema as a string constant for runtime JSON parsing.

set(GENERATED_DIR ${PROJECT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

set(SCHEMA_FILES
    ${PROJECT_SOURCE_DIR}/schemas/simulation_config.fbs
    ${PROJECT_SOURCE_DIR}/schemas/checkpoint.fbs
)

set(GENERATED_HEADERS)
foreach(schema ${SCHEMA_FILES})
    get_filename_component(schema_name ${schema} NAME_WE)
    set(output_header "${GENERATED_DIR}/${schema_name}_generated.h")
    add_custom_command(
        OUTPUT ${output_header}
        COMMAND
            $<TARGET_FILE:flatc> --cpp -I ${PROJECT_SOURCE_DIR}/schemas -o
            ${GENERATED_DIR} ${schema}
        DEPENDS flatc ${schema}
        COMMENT "Compiling FlatBuffers schema: ${schema_name}.fbs"
    )
    list(APPEND GENERATED_HEADERS ${output_header})
endforeach()

add_custom_target(flatbuffers_schemas DEPENDS ${GENERATED_HEADERS})

# Embed simulation_config.fbs as a C++ string constant so the library can parse
# JSON configs without locating the schema file at runtime.
file(READ "${PROJECT_SOURCE_DIR}/schemas/simulation_config.fbs" _schema_content)
file(
    WRITE
    "${GENERATED_DIR}/embedded_schema.h"
    "// Auto-generated from simulation_config.fbs — do not edit.\n"
    "#pragma once\n"
    "namespace rocjitsu {\n"
    "inline constexpr const char kEmbeddedSchema[] = R\"rjschema(\n"
    "${_schema_content}"
    ")rjschema\";\n"
    "} // namespace rocjitsu\n"
)
