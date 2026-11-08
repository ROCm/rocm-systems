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

    # The application calls the range API (begin/end, or the scoped host macros). It does
    # not link librocprofiler-sdk: that link makes the first kernel launch fail on gfx942
    # (see samples/kernel_replay/CMakeLists.txt). The preloaded tool brings the SDK in,
    # and the link leaves the API references unresolved so that library satisfies them.
    target_link_libraries(
        ${NAME} PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
                        rocprofiler-sdk::samples-build-flags Threads::Threads)
    target_link_options(${NAME} PRIVATE
                        "LINKER:--unresolved-symbols=ignore-in-object-files")
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

# Divergence checking is what makes this sample a check on repeatability rather than just
# on the callback sequence. It compares the state the application's own execution left
# behind against the state after the final replayed pass, so a between-pass rewind that
# did not happen, or a recording replayed out of order, shows up as a non-zero divergence
# count at CLOSE. Without it the range-exit restore would hide both.
add_range_replay_sample(range-replay-basic basic_client.cpp ENVIRONMENT
                        "ROCPROF_RANGE_REPLAY_VERIFY=1")

# Same checks as range-replay-basic, but the extra passes re-enter the range through
# siglongjmp instead of re-submitting the recorded packets. RR_APP_MODE=host opens the
# range with ROCPROFILER_REPLAY_RANGE_BEGIN; RR_HOST_REPLAY tells the tool to request that
# mode.
add_range_replay_sample(
    range-replay-host basic_client.cpp ENVIRONMENT "ROCPROF_RANGE_REPLAY_VERIFY=1"
    "RR_APP_MODE=host" "RR_HOST_REPLAY=1")
add_range_replay_sample(range-replay-opt-out opt_out_client.cpp)

# GPU_MAX_HW_QUEUES is what makes the decline deterministic. HIP pools hardware queues and
# round-robins streams onto them, so with a pool of one the sample's two streams would
# share a queue and the range would be replayed rather than declined. The default is 4,
# but pinning it means the sample does not depend on that default.
add_range_replay_sample(range-replay-decline decline_client.cpp ENVIRONMENT
                        "RR_APP_MODE=multi-queue" "GPU_MAX_HW_QUEUES=4")
