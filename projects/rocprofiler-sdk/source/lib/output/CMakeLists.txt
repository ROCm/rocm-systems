#
# Tool library used by rocprofiler
#

rocprofiler_activate_clang_tidy()

set(TOOL_OUTPUT_HEADERS
    agent_info.hpp
    buffered_output.hpp
    counter_info.hpp
    csv.hpp
    csv_output_file.hpp
    domain_type.hpp
    format_path.hpp
    generateCSV.hpp
    generateJSON.hpp
    generateOTF2.hpp
    generatePerfetto.hpp
    generateStats.hpp
    generateRocpd.hpp
    generator.hpp
    kernel_symbol_info.hpp
    host_symbol_info.hpp
    metadata.hpp
    node_info.hpp
    output_config.hpp
    output_key.hpp
    output_stream.hpp
    statistics.hpp
    stream_info.hpp
    timestamps.hpp
    tmp_file_buffer.hpp
    tmp_file.hpp)

set(TOOL_OUTPUT_SOURCES
    csv_output_file.cpp
    counter_info.cpp
    domain_type.cpp
    format_path.cpp
    generateCSV.cpp
    generateJSON.cpp
    generateOTF2.cpp
    generatePerfetto.cpp
    generateStats.cpp
    generateRocpd.cpp
    metadata.cpp
    node_info.cpp
    output_config.cpp
    output_key.cpp
    output_stream.cpp
    statistics.cpp
    tmp_file_buffer.cpp
    tmp_file.cpp)

add_library(rocprofiler-sdk-output-library STATIC)
add_library(rocprofiler-sdk::rocprofiler-sdk-output-library ALIAS
            rocprofiler-sdk-output-library)
target_sources(rocprofiler-sdk-output-library PRIVATE ${TOOL_OUTPUT_SOURCES}
                                                      ${TOOL_OUTPUT_HEADERS})
target_link_libraries(
    rocprofiler-sdk-output-library
    PUBLIC rocprofiler-sdk::rocprofiler-sdk-rocpd-library
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-memcheck
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-cereal
            rocprofiler-sdk::rocprofiler-sdk-perfetto
            rocprofiler-sdk::rocprofiler-sdk-otf2
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
            rocprofiler-sdk::rocprofiler-sdk-dw
            rocprofiler-sdk::rocprofiler-sdk-elf
            rocprofiler-sdk::rocprofiler-sdk-sqlite3)

target_compile_definitions(rocprofiler-sdk-output-library
                           PRIVATE PROJECT_BINARY_DIR="${PROJECT_BINARY_DIR}")

add_subdirectory(sql)
