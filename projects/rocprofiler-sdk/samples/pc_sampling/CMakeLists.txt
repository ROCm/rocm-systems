#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-samples-pc-sampling LANGUAGES C CXX)

if(NOT TARGET rocprofiler::rocprofiler-library)
    find_package(rocprofiler REQUIRED)
endif()

add_executable(pc_sampling_single-user-host-trap)
target_sources(pc_sampling_single-user-host-trap PRIVATE common.h
                                                         single-user-host-trap.cpp)
target_link_libraries(
    pc_sampling_single-user-host-trap
    PRIVATE rocprofiler::rocprofiler-library
            $<TARGET_NAME_IF_EXISTS:rocprofiler::samples-build-flags>)

add_executable(pc_sampling_single-user-host-trap-retry)
target_sources(pc_sampling_single-user-host-trap-retry
               PRIVATE common.h single-user-host-trap-retries-service-instantiation.cpp)
target_link_libraries(
    pc_sampling_single-user-host-trap-retry
    PRIVATE rocprofiler::rocprofiler-library
            $<TARGET_NAME_IF_EXISTS:rocprofiler::samples-build-flags>)

add_executable(pc_sampling_single-user-multiple-agents)
target_sources(pc_sampling_single-user-multiple-agents
               PRIVATE common.h single-user-multiple-agents.cpp)
target_link_libraries(
    pc_sampling_single-user-multiple-agents
    PRIVATE rocprofiler::rocprofiler-library
            $<TARGET_NAME_IF_EXISTS:rocprofiler::samples-build-flags>)
