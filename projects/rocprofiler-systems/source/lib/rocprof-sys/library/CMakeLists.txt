#
set(library_sources
    ${CMAKE_CURRENT_LIST_DIR}/coverage.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cpu_freq.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kokkosp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ompt.cpp
    ${CMAKE_CURRENT_LIST_DIR}/perf.cpp
    ${CMAKE_CURRENT_LIST_DIR}/process_sampler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ptl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sampling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/thread_deleter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/thread_info.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracing.cpp
)

set(library_headers
    ${CMAKE_CURRENT_LIST_DIR}/coverage.hpp
    ${CMAKE_CURRENT_LIST_DIR}/cpu_freq.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ompt.hpp
    ${CMAKE_CURRENT_LIST_DIR}/process_sampler.hpp
    ${CMAKE_CURRENT_LIST_DIR}/perf.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ptl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/rocm.hpp
    ${CMAKE_CURRENT_LIST_DIR}/amd_smi.hpp
    ${CMAKE_CURRENT_LIST_DIR}/rocprofiler-sdk.hpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sampling.hpp
    ${CMAKE_CURRENT_LIST_DIR}/thread_data.hpp
    ${CMAKE_CURRENT_LIST_DIR}/thread_deleter.hpp
    ${CMAKE_CURRENT_LIST_DIR}/thread_info.hpp
    ${CMAKE_CURRENT_LIST_DIR}/tracing.hpp
)

target_sources(
    rocprofiler-systems-object-library
    PRIVATE ${library_sources} ${library_headers}
)

if(ROCPROFSYS_USE_ROCM)
    target_sources(
        rocprofiler-systems-object-library
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/rocm.cpp
            ${CMAKE_CURRENT_LIST_DIR}/rocprofiler-sdk.cpp
            ${CMAKE_CURRENT_LIST_DIR}/amd_smi.cpp
    )
    add_subdirectory(rocprofiler-sdk)
endif()

add_subdirectory(causal)
add_subdirectory(components)
add_subdirectory(coverage)
add_subdirectory(tracing)

set(ndebug_sources
    ${CMAKE_CURRENT_LIST_DIR}/components/mpi_gotcha.cpp
    ${CMAKE_CURRENT_LIST_DIR}/components/backtrace_metrics.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kokkosp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/amd_smi.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ompt.cpp
)

set_source_files_properties(
    ${ndebug_sources}
    DIRECTORY ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
    PROPERTIES COMPILE_DEFINITIONS NDEBUG COMPILE_OPTIONS "-g0;-O3"
)
