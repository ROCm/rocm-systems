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

project(rocprofiler-sdk-samples-range-replay LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

# The `headers` component is what lets the application declare its range boundaries
# against the real types without linking the SDK; see the target_link_libraries comment
# below.
find_package(rocprofiler-sdk REQUIRED COMPONENTS headers shared-library)
find_package(Threads REQUIRED)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

function(add_range_replay_sample NAME CLIENT_SRC)
    cmake_parse_arguments(arg "" "" "ENVIRONMENT" ${ARGN})

    add_library(${NAME}-client SHARED)
    target_sources(${NAME}-client PRIVATE ${CLIENT_SRC} client.hpp range.hpp)
    target_link_libraries(
        ${NAME}-client
        PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-build-flags
                rocprofiler-sdk::samples-common-library)

    add_executable(${NAME})
    target_sources(${NAME} PRIVATE main.cpp range.hpp)

    # Unlike the kernel replay samples, this application calls the SDK: range boundaries
    # are the tool's to declare but the application's to place. It resolves the two range
    # entry points with dlsym out of the preloaded client and links only the headers,
    # because linking the SDK into the HIP executable makes HIP report
    # hipErrorInvalidDeviceFunction on the first kernel launch on gfx942 (see
    # samples/kernel_replay/CMakeLists.txt).
    target_link_libraries(
        ${NAME}
        PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
                rocprofiler-sdk::samples-build-flags Threads::Threads ${CMAKE_DL_LIBS})
    add_dependencies(${NAME} ${NAME}-client)

    rocprofiler_samples_get_preload_env(PRELOAD_ENV ${NAME}-client)
    rocprofiler_samples_get_ld_library_path_env(LIBRARY_PATH_ENV ${NAME}-client)
    set(_env ${PRELOAD_ENV} ${LIBRARY_PATH_ENV} ${arg_ENVIRONMENT})

    add_test(NAME ${NAME} COMMAND $<TARGET_FILE:${NAME}>)

    # These are the range-replay feature samples; do not hide them behind the repo-wide
    # unstable gate.
    set_tests_properties(
        ${NAME}
        PROPERTIES TIMEOUT 120 LABELS "samples;range-replay" ENVIRONMENT "${_env}"
                   FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endfunction()

add_range_replay_sample(range-replay-basic basic_client.cpp)
add_range_replay_sample(range-replay-opt-out opt_out_client.cpp)
add_range_replay_sample(range-replay-decline decline_client.cpp ENVIRONMENT
                        "RR_APP_MODE=multi-queue")
