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
    host_symbol_info.hpp
    kernel_symbol_info.hpp
    kfd_info.hpp
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
    generateStats.cpp
    metadata.cpp
    output_config.cpp
    output_key.cpp
    output_stream.cpp
    statistics.cpp
    tmp_file_buffer.cpp
    tmp_file.cpp)

if(WIN32)
    # uname and sysfs have no Windows counterpart; the same fields come from the registry.
    list(APPEND TOOL_OUTPUT_SOURCES node_info_windows.cpp)
else()
    list(APPEND TOOL_OUTPUT_SOURCES node_info.cpp)
endif()

# The remaining generators are excluded on Windows: perfetto, otf2 and rocpd/sqlite3 are
# not built here, and the JSON generator serializes the code object and PC sampling data
# that the ETW tracing path does not produce.
if(NOT WIN32)
    list(APPEND TOOL_OUTPUT_SOURCES generateJSON.cpp generateOTF2.cpp
         generatePerfetto.cpp generateRocpd.cpp)
endif()

add_library(rocprofiler-sdk-output-library STATIC)
add_library(rocprofiler-sdk::rocprofiler-sdk-output-library ALIAS
            rocprofiler-sdk-output-library)
target_sources(rocprofiler-sdk-output-library PRIVATE ${TOOL_OUTPUT_SOURCES}
                                                      ${TOOL_OUTPUT_HEADERS})
# targets linking to this library should also link to the
# rocprofiler-sdk::rocprofiler-sdk-sqlite3 target if the target is a shared library. Do
# not link to the sqlite3 library here because the Python bindings target should not
# directly link to the sqlite3 library.
target_link_libraries(
    rocprofiler-sdk-output-library
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-sanitizer
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-cereal)

# Everything below backs a generator that is not built on Windows, except comgr/dw/elf
# which back the code object disassembly that metadata.cpp only compiles in elsewhere.
if(NOT WIN32)
    target_link_libraries(
        rocprofiler-sdk-output-library
        PUBLIC rocprofiler-sdk::rocprofiler-sdk-rocpd-library
               rocprofiler-sdk::rocprofiler-sdk-aqlprofile
        PRIVATE rocprofiler-sdk::rocprofiler-sdk-perfetto
                rocprofiler-sdk::rocprofiler-sdk-otf2
                rocprofiler-sdk::rocprofiler-sdk-amd-comgr
                rocprofiler-sdk::rocprofiler-sdk-dw
                rocprofiler-sdk::rocprofiler-sdk-elf)
endif()

target_compile_definitions(rocprofiler-sdk-output-library
                           PRIVATE PROJECT_BINARY_DIR="${PROJECT_BINARY_DIR}")

# sql/ is the rocpd backing store and needs sqlite3
if(NOT WIN32)
    add_subdirectory(sql)
endif()
