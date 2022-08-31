#
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/defines.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/defines.hpp @ONLY)

set(library_sources
    ${CMAKE_CURRENT_LIST_DIR}/config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/coverage.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cpu_freq.cpp
    ${CMAKE_CURRENT_LIST_DIR}/critical_trace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/debug.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dynamic_library.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kokkosp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mproc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ompt.cpp
    ${CMAKE_CURRENT_LIST_DIR}/perfetto.cpp
    ${CMAKE_CURRENT_LIST_DIR}/process_sampler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ptl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sampling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/state.cpp
    ${CMAKE_CURRENT_LIST_DIR}/thread_data.cpp
    ${CMAKE_CURRENT_LIST_DIR}/thread_info.cpp
    ${CMAKE_CURRENT_LIST_DIR}/timemory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracing.cpp)

set(library_headers
    ${CMAKE_CURRENT_LIST_DIR}/categories.hpp
    ${CMAKE_CURRENT_LIST_DIR}/config.hpp
    ${CMAKE_CURRENT_LIST_DIR}/common.hpp
    ${CMAKE_CURRENT_LIST_DIR}/concepts.hpp
    ${CMAKE_CURRENT_LIST_DIR}/coverage.hpp
    ${CMAKE_CURRENT_LIST_DIR}/cpu_freq.hpp
    ${CMAKE_CURRENT_LIST_DIR}/critical_trace.hpp
    ${CMAKE_CURRENT_LIST_DIR}/debug.hpp
    ${CMAKE_CURRENT_LIST_DIR}/dynamic_library.hpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu.hpp
    ${CMAKE_CURRENT_LIST_DIR}/mproc.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ompt.hpp
    ${CMAKE_CURRENT_LIST_DIR}/perfetto.hpp
    ${CMAKE_CURRENT_LIST_DIR}/process_sampler.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ptl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/rcclp.hpp
    ${CMAKE_CURRENT_LIST_DIR}/rocm.hpp
    ${CMAKE_CURRENT_LIST_DIR}/rocm_smi.hpp
    ${CMAKE_CURRENT_LIST_DIR}/rocprofiler.hpp
    ${CMAKE_CURRENT_LIST_DIR}/roctracer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/runtime.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sampling.hpp
    ${CMAKE_CURRENT_LIST_DIR}/state.hpp
    ${CMAKE_CURRENT_LIST_DIR}/thread_data.hpp
    ${CMAKE_CURRENT_LIST_DIR}/thread_info.hpp
    ${CMAKE_CURRENT_LIST_DIR}/timemory.hpp
    ${CMAKE_CURRENT_LIST_DIR}/tracing.hpp
    ${CMAKE_CURRENT_LIST_DIR}/utility.hpp)

target_sources(omnitrace-object-library PRIVATE ${library_sources} ${library_headers}
                                                ${CMAKE_CURRENT_BINARY_DIR}/defines.hpp)

if(OMNITRACE_USE_ROCTRACER OR OMNITRACE_USE_ROCPROFILER)
    target_sources(
        omnitrace-object-library PRIVATE ${CMAKE_CURRENT_LIST_DIR}/rocprofiler.cpp
                                         ${CMAKE_CURRENT_LIST_DIR}/rocm.cpp)
endif()

if(OMNITRACE_USE_ROCTRACER)
    target_sources(omnitrace-object-library
                   PRIVATE ${CMAKE_CURRENT_LIST_DIR}/roctracer.cpp)
endif()

if(OMNITRACE_USE_RCCL)
    target_sources(omnitrace-object-library PRIVATE ${CMAKE_CURRENT_LIST_DIR}/rcclp.cpp)
endif()

if(OMNITRACE_USE_ROCPROFILER)
    target_sources(
        omnitrace-object-library PRIVATE ${CMAKE_CURRENT_LIST_DIR}/rocprofiler.cpp
                                         ${CMAKE_CURRENT_LIST_DIR}/rocprofiler.hpp)
endif()

if(OMNITRACE_USE_ROCM_SMI)
    target_sources(omnitrace-object-library
                   PRIVATE ${CMAKE_CURRENT_LIST_DIR}/rocm_smi.cpp)
endif()

add_subdirectory(components)
add_subdirectory(rocprofiler)
