# Perfetto trace_processor POC
# Opt-in; only reached when PROFILER_HUB_ENABLE_TPX=ON.
# Requires:
#   PERFETTO_SOURCE_DIR  — root of a google/perfetto checkout
#   PERFETTO_BUILD_DIR   — GN out/ directory containing libtrace_processor.a

cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED PERFETTO_SOURCE_DIR OR NOT DEFINED PERFETTO_BUILD_DIR)
    message(
        FATAL_ERROR
        "PROFILER_HUB_ENABLE_TPX requires -DPERFETTO_SOURCE_DIR=<path> "
        "and -DPERFETTO_BUILD_DIR=<path>"
    )
endif()

set(TPX_LIB "${PERFETTO_BUILD_DIR}/libtrace_processor.a")
if(NOT EXISTS "${TPX_LIB}")
    message(
        FATAL_ERROR
        "libtrace_processor.a not found at ${TPX_LIB}. "
        "Build it first: ninja -C ${PERFETTO_BUILD_DIR} src/trace_processor:trace_processor"
    )
endif()

add_library(perfetto_trace_processor STATIC IMPORTED)
set_target_properties(
    perfetto_trace_processor
    PROPERTIES
        IMPORTED_LOCATION "${TPX_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES
            "${PERFETTO_SOURCE_DIR}/include;${PERFETTO_BUILD_DIR}/gen;${PERFETTO_BUILD_DIR}/gen/build_config"
)

add_executable(tpx_poc tpx_poc.cpp)
target_link_libraries(tpx_poc PRIVATE perfetto_trace_processor pthread dl)
target_compile_features(tpx_poc PRIVATE cxx_std_17)
