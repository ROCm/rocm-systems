# ############################################################################################################################################
# ROCProfiler Tool
# ############################################################################################################################################
# Setting Default Binary output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Getting Source files for ROCProfiler Tool
file(GLOB ROCPROFILER_TOOL_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/tool.cpp)
file(GLOB ROCPROFILER_UTIL_SRC_FILES ${PROJECT_SOURCE_DIR}/src/utils/helper.cpp)

# Compiling/Installing ROCProfiler API
add_library(rocprofiler_tool SHARED ${ROCPROFILER_TOOL_SRC_FILES}
                                    ${ROCPROFILER_UTIL_SRC_FILES})

set_target_properties(
    rocprofiler_tool
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/rocprofiler
               LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/exportmap
               INSTALL_RPATH "${ROCM_APPEND_PRIVLIB_RPATH}")

target_include_directories(
    rocprofiler_tool PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
                             ${PROJECT_SOURCE_DIR}/src)

target_compile_definitions(
    rocprofiler_tool
    PUBLIC AMD_INTERNAL_BUILD
    PRIVATE HIP_PROF_HIP_API_STRING=1 __HIP_PLATFORM_AMD__=1)

target_link_libraries(rocprofiler_tool
                      PRIVATE $<BUILD_INTERFACE:rocprofiler::build-flags>)

target_link_libraries(
    rocprofiler_tool
    PUBLIC rocprofiler-v2 hsa-runtime64::hsa-runtime64 Threads::Threads atomic dl rt
           stdc++fs amd_comgr
    PRIVATE rocprofiler::memcheck)
target_link_options(
    rocprofiler_tool PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exportmap
    -Wl,--no-undefined)

install(TARGETS rocprofiler_tool LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/rocprofiler
                                         COMPONENT runtime)
install(TARGETS rocprofiler_tool LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/rocprofiler
                                         COMPONENT asan)

add_subdirectory(rocsys)
#add_subdirectory(rocprofv2)
add_subdirectory(versioning)

add_executable(ctrl ctrl.cpp)
set_target_properties(
    ctrl PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/libexec/rocprofiler
                    INSTALL_RPATH "$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}")
target_link_options(
    rocprofiler_tool PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exportmap
    -Wl,--no-undefined)
target_link_libraries(ctrl PRIVATE rocprofiler-v2 hsa-runtime64::hsa-runtime64)
install(TARGETS ctrl RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/rocprofiler
                             COMPONENT runtime)

# ########################################################################################
