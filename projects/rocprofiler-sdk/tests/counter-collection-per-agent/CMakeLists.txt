#
#
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

project(
    rocprofiler-sdk-tests-counter-collection-per-agent
    LANGUAGES CXX HIP
    VERSION 0.0.0)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

find_package(rocprofiler-sdk REQUIRED)
find_package(Threads REQUIRED)

# The tool and the workload live in one executable so the test can start and stop contexts
# around the measured region instead of profiling the whole process.
set_source_files_properties(per_agent_scoping_app.cpp PROPERTIES LANGUAGE HIP)
add_executable(counter-collection-per-agent-app)
target_sources(counter-collection-per-agent-app PRIVATE per_agent_scoping_app.cpp)
target_link_libraries(counter-collection-per-agent-app
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk Threads::Threads)

# Reports SKIP and exits 0 on a machine with fewer than two GPUs.
rocprofiler_add_integration_execute_test(
    counter-collection-per-agent
    TARGET counter-collection-per-agent-app
    TIMEOUT 120
    LABELS "integration-tests"
    SKIP_REGULAR_EXPRESSION "SKIP: per-agent scoping requires at least 2 GPUs")
