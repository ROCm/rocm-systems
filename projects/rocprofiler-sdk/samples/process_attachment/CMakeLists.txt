#
# CMakeLists.txt for process attachment sample
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

project(rocprofiler-sdk-samples-process-attachment LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)

# 1. Process Attachment Tool Library
# This is the tool that gets dynamically loaded into target processes
add_library(process-attachment-tool SHARED)
target_sources(process-attachment-tool PRIVATE 
    tool.cpp 
    tool.hpp
)
target_link_libraries(
    process-attachment-tool
    PRIVATE 
        rocprofiler-sdk::rocprofiler-sdk 
        rocprofiler-sdk::samples-build-flags
        rocprofiler-sdk::samples-common-library
)

# Make sure the tool library can be found for attachment
set_target_properties(process-attachment-tool PROPERTIES
    OUTPUT_NAME "process-attachment-tool"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# 2. Process Attachment Client
# This is the utility that performs the attachment to running processes
find_package(Threads REQUIRED)

add_executable(process-attachment-client)
target_sources(process-attachment-client PRIVATE client.cpp)
target_link_libraries(
    process-attachment-client
    PRIVATE
        Threads::Threads
        ${CMAKE_DL_LIBS}
        rocprofiler-sdk::samples-build-flags
)

# 3. Target Application
# This is a sample long-running application that can be attached to
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
find_package(rocprofiler-sdk-roctx REQUIRED)

add_executable(process-attachment-target)
target_sources(process-attachment-target PRIVATE main.cpp)
target_link_libraries(
    process-attachment-target
    PRIVATE 
        Threads::Threads
        rocprofiler-sdk-roctx::rocprofiler-sdk-roctx
        rocprofiler-sdk::samples-build-flags
)

# Get environment for running tests
rocprofiler_samples_get_preload_env(PRELOAD_ENV process-attachment-tool)
rocprofiler_samples_get_ld_library_path_env(
    LIBRARY_PATH_ENV 
    rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library
)

set(process-attachment-env ${PRELOAD_ENV} ${LIBRARY_PATH_ENV})

# Test that verifies the target application runs correctly
add_test(NAME process-attachment-target-test 
         COMMAND $<TARGET_FILE:process-attachment-target> -t 5)

set_tests_properties(
    process-attachment-target-test
    PROPERTIES 
        TIMEOUT 30 
        LABELS "samples" 
        ENVIRONMENT "${process-attachment-env}"
        FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
)

# Integration test that runs attachment scenario
# This test is more complex and may be unstable in some environments
if(NOT ROCPROFILER_DISABLE_UNSTABLE_CTESTS)
    add_test(NAME process-attachment-integration-test 
             COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_attachment.sh 
                     $<TARGET_FILE:process-attachment-target>
                     $<TARGET_FILE:process-attachment-client>
                     ${CMAKE_CURRENT_BINARY_DIR})

    set_tests_properties(
        process-attachment-integration-test
        PROPERTIES 
            TIMEOUT 60 
            LABELS "samples;integration" 
            ENVIRONMENT "${process-attachment-env}"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

# Install targets
install(
    TARGETS process-attachment-tool process-attachment-client process-attachment-target
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT samples
)
