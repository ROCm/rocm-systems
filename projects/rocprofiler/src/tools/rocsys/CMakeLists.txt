# ############################################################################################################################################
# ROCProfiler ROCSYS Frontend
# ############################################################################################################################################
set(CMAKE_BINARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Getting Source files for RDC, Utils, Counters
file(GLOB ROCPROFILER_ROCSYS_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Compiling/Installing ROCProfiler API
add_executable(rocprofiler_rocsys_fe ${ROCPROFILER_ROCSYS_SRC_FILES})

set_target_properties(rocprofiler_rocsys_fe PROPERTIES OUTPUT_NAME "rocsys")

target_include_directories(
    rocprofiler_rocsys_fe PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
                                  ${PROJECT_SOURCE_DIR}/inc)

target_link_libraries(rocprofiler_rocsys_fe dl rt stdc++fs)

install(
    TARGETS rocprofiler_rocsys_fe
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT runtime)

# ########################################################################################
