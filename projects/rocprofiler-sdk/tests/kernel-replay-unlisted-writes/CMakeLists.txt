#
# Kernel replay of a kernel that writes a different tile in every pass. The client tool replays it
# with four passes and hash-once restores, so passes after the first write tiles off the restore
# list kept from the first. The check before the last pass has to find them and restore everything,
# so the memory the application keeps is that of one run of the last pass.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

project(rocprofiler-sdk-tests-kernel-replay-unlisted-writes LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)

add_library(kernel-replay-unlisted-writes-client SHARED)
target_sources(kernel-replay-unlisted-writes-client PRIVATE client.cpp)
target_link_libraries(kernel-replay-unlisted-writes-client
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(kernel-replay-unlisted-writes-testapp)
target_sources(kernel-replay-unlisted-writes-testapp PRIVATE main.cpp)
target_link_libraries(kernel-replay-unlisted-writes-testapp PRIVATE ${CMAKE_DL_LIBS})

rocprofiler_add_integration_execute_test(
    test-kernel-replay-unlisted-writes
    COMMAND $<TARGET_FILE:kernel-replay-unlisted-writes-testapp>
    DEPENDS kernel-replay-unlisted-writes-client
    TIMEOUT 60
    LABELS "integration-tests;kernel-replay"
    PRELOAD "$<TARGET_FILE:kernel-replay-unlisted-writes-client>"
    ENVIRONMENT "ROCPROFILER_INTERNAL_REPLAY_RESTORE_FORCE_MODE=hash-once")
