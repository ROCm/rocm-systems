#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-tests-spm
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

set(COUNTER_LIST SQ_CYCLES SQ_WAVES TCC_HIT TCC_MISS TCC_REQ CPC_CPC_STAT_BUSY
                 CPC_CPC_STAT_IDLE)

set(IS_DISABLED True)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

add_subdirectory(json)
add_subdirectory(rocpd)
