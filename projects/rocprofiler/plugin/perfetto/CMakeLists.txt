rocprofiler_checkout_git_submodule(
            RECURSIVE
            TEST_FILE "sdk"
            RELATIVE_PATH perfetto
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            REPO_URL https://github.com/google/perfetto.git
            REPO_BRANCH "releases/v44.x")

include_directories(perfetto/sdk)

file(GLOB ROCPROFILER_UTIL_SRC_FILES ${PROJECT_SOURCE_DIR}/src/utils/helper.cpp)

add_library(perfetto_plugin ${LIBRARY_TYPE} ${ROCPROFILER_UTIL_SRC_FILES} perfetto/sdk/perfetto.cc perfetto.cpp)

set_target_properties(
    perfetto_plugin
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../exportmap
               LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/rocprofiler
               INSTALL_RPATH "${ROCM_APPEND_PRIVLIB_RPATH}")

target_compile_definitions(perfetto_plugin PRIVATE HIP_PROF_HIP_API_STRING=1
                                                   __HIP_PLATFORM_AMD__=1)

target_include_directories(perfetto_plugin PRIVATE ${PROJECT_SOURCE_DIR})

target_link_options(
    perfetto_plugin PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exportmap
    -Wl,--no-undefined)

target_link_libraries(perfetto_plugin PRIVATE rocprofiler-v2
                                              Threads::Threads stdc++fs amd_comgr)

install(TARGETS perfetto_plugin
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME} COMPONENT plugins)
