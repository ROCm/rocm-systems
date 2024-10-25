#
# Tool library used by rocprofiler
#

rocprofiler_activate_clang_tidy()

set(TOOL_HEADERS
    buffered_output.hpp
    config.hpp
    csv.hpp
    domain_type.hpp
    generateCSV.hpp
    generateJSON.hpp
    generateOTF2.hpp
    generatePerfetto.hpp
    generateStats.hpp
    helper.hpp
    output_file.hpp
    statistics.hpp
    tmp_file_buffer.hpp
    tmp_file.hpp)

set(TOOL_SOURCES
    config.cpp
    domain_type.cpp
    generateCSV.cpp
    generateJSON.cpp
    generateOTF2.cpp
    generatePerfetto.cpp
    generateStats.cpp
    helper.cpp
    main.c
    output_file.cpp
    statistics.cpp
    tmp_file_buffer.cpp
    tmp_file.cpp
    tool.cpp)

add_library(rocprofiler-sdk-tool SHARED)
target_sources(rocprofiler-sdk-tool PRIVATE ${TOOL_SOURCES} ${TOOL_HEADERS})

target_link_libraries(
    rocprofiler-sdk-tool
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-shared-library
            rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-memcheck
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-cereal
            rocprofiler-sdk::rocprofiler-sdk-perfetto
            rocprofiler-sdk::rocprofiler-sdk-otf2)

set_target_properties(
    rocprofiler-sdk-tool
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY
               ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               BUILD_RPATH "\$ORIGIN:\$ORIGIN/.."
               INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")

install(
    TARGETS rocprofiler-sdk-tool
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk
    COMPONENT tools
    EXPORT rocprofiler-sdk-tool-targets)

add_subdirectory(kokkosp)
