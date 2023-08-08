set(common_sources ${CMAKE_CURRENT_LIST_DIR}/config.cpp
                   ${CMAKE_CURRENT_LIST_DIR}/helper.cpp)

set(common_headers
    ${CMAKE_CURRENT_LIST_DIR}/config.hpp ${CMAKE_CURRENT_LIST_DIR}/defines.hpp
    ${CMAKE_CURRENT_LIST_DIR}/environment.hpp ${CMAKE_CURRENT_LIST_DIR}/join.hpp
    ${CMAKE_CURRENT_LIST_DIR}/log.hpp ${CMAKE_CURRENT_LIST_DIR}/helper.hpp)

add_library(rocprofiler-common-library STATIC)
add_library(rocprofiler::rocprofiler-common-library ALIAS rocprofiler-common-library)

add_subdirectory(container)

target_sources(rocprofiler-common-library PRIVATE ${common_sources} ${common_headers})
target_include_directories(rocprofiler-common-library
                           PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)

target_link_libraries(
    rocprofiler-common-library
    PUBLIC rocprofiler::rocprofiler-amd-comgr
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-build-flags>
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-memcheck>
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-stdcxxfs>
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-dl>)
set_target_properties(rocprofiler-common-library PROPERTIES OUTPUT_NAME
                                                            rocprofiler-common)
