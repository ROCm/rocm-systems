include_directories(json/include)

file(GLOB ROCPROFILER_UTIL_SRC_FILES ${PROJECT_SOURCE_DIR}/src/utils/helper.cpp)
file(GLOB SPEEDSCOPE_HEADER_FILES
     ${CMAKE_CURRENT_SOURCE_DIR}/json/include/nlohmann/json.hpp)

add_library(json_plugin ${LIBRARY_TYPE} ${ROCPROFILER_UTIL_SRC_FILES}
                        ${SPEEDSCOPE_HEADER_FILES} json.cpp)

set_target_properties(
    json_plugin
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../exportmap
               LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/rocprofiler
               INSTALL_RPATH "${ROCM_APPEND_PRIVLIB_RPATH}")

target_compile_definitions(json_plugin PRIVATE HIP_PROF_HIP_API_STRING=1
                                               __HIP_PLATFORM_AMD__=1)

target_include_directories(json_plugin PRIVATE ${PROJECT_SOURCE_DIR})

target_link_options(
    json_plugin PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exportmap
    -Wl,--no-undefined)

target_link_libraries(json_plugin PRIVATE rocprofiler-v2 Threads::Threads stdc++fs
                                          amd_comgr)

install(TARGETS json_plugin LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
                                    COMPONENT plugins)
