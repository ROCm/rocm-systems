#
# Range replay against concurrent work on the same agent.
#
# main.cpp replays ranges on its main thread while a second thread writes its own device buffer
# just after each end() call, by async copy or by kernel, so the write lands while the range's
# replay window runs. The application fails if a replay window reverted a write; the test timeout
# catches a window that deadlocks against the second thread's submission. The tool checks that
# every range closed exactly once, replayed or declined for the concurrent write. The
# single-queue variants make HIP put both threads' streams on one hardware queue.
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

project(rocprofiler-sdk-tests-range-replay-concurrency LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

find_package(rocprofiler-sdk REQUIRED COMPONENTS headers shared-library)
find_package(Threads REQUIRED)

add_library(range-replay-concurrency-client SHARED)
target_sources(range-replay-concurrency-client PRIVATE client.cpp range.hpp)
target_include_directories(range-replay-concurrency-client
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-concurrency-client PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                            rocprofiler-sdk::tests-build-flags)

# The application resolves the range replay entry points with dlsym instead of linking the
# SDK, for the same reason as the range replay samples (linking rocprofiler-sdk into a HIP
# executable breaks the first launch on gfx942).
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(range-replay-concurrency-testapp)
target_sources(range-replay-concurrency-testapp PRIVATE main.cpp range.hpp)
target_include_directories(range-replay-concurrency-testapp
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-concurrency-testapp
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers rocprofiler-sdk::tests-build-flags
            Threads::Threads ${CMAKE_DL_LIBS})
add_dependencies(range-replay-concurrency-testapp range-replay-concurrency-client)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}:$<TARGET_FILE:range-replay-concurrency-client>"
        )
else()
    set(PRELOAD_ENV "$<TARGET_FILE:range-replay-concurrency-client>")
endif()

function(add_rr_concurrency_test NAME)
    cmake_parse_arguments(arg "" "" "ENVIRONMENT" ${ARGN})
    rocprofiler_add_integration_execute_test(
        ${NAME}
        COMMAND $<TARGET_FILE:range-replay-concurrency-testapp>
        DEPENDS range-replay-concurrency-client
        TIMEOUT 180
        LABELS "integration-tests;range-replay"
        PRELOAD "${PRELOAD_ENV}"
        ENVIRONMENT "RR_CONC_RANGES=16" ${arg_ENVIRONMENT}
        PASS_REGULAR_EXPRESSION "\\[rr-conc\\] PASS"
        FAIL_REGULAR_EXPRESSION
            "\\[rr-conc\\] FAIL|\\[rr-conc\\] error|\\[app\\] FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        )
endfunction()

foreach(_MODE packet host)
    foreach(_WRITER copy kernel)
        add_rr_concurrency_test(
            test-range-replay-concurrency-${_MODE}-${_WRITER} ENVIRONMENT
            "RR_CONC_MODE=${_MODE};RR_CONC_WRITER=${_WRITER}")
        add_rr_concurrency_test(
            test-range-replay-concurrency-${_MODE}-${_WRITER}-one-queue ENVIRONMENT
            "RR_CONC_MODE=${_MODE};RR_CONC_WRITER=${_WRITER};GPU_MAX_HW_QUEUES=1")
    endforeach()
endforeach()
