# ------------------------------------------------------------------------------#
#
# omnitrace interface library
#
# ------------------------------------------------------------------------------#

add_library(omnitrace-interface-library INTERFACE)
add_library(omnitrace::omnitrace-interface-library ALIAS omnitrace-interface-library)

target_include_directories(
    omnitrace-interface-library INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                                          ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(
    omnitrace-interface-library
    INTERFACE
        OMNITRACE_MAX_THREADS=${OMNITRACE_MAX_THREADS}
        $<BUILD_INTERFACE:$<IF:$<BOOL:${OMNITRACE_CUSTOM_DATA_SOURCE}>,CUSTOM_DATA_SOURCE,>>
    )

target_link_libraries(
    omnitrace-interface-library
    INTERFACE
        $<BUILD_INTERFACE:omnitrace::omnitrace-headers>
        $<BUILD_INTERFACE:omnitrace::omnitrace-threading>
        $<BUILD_INTERFACE:omnitrace::omnitrace-common-library>
        $<BUILD_INTERFACE:omnitrace::omnitrace-compile-options>
        $<BUILD_INTERFACE:omnitrace::omnitrace-compile-definitions>
        $<BUILD_INTERFACE:omnitrace::omnitrace-perfetto>
        $<BUILD_INTERFACE:omnitrace::omnitrace-timemory>
        $<BUILD_INTERFACE:omnitrace::omnitrace-mpi>
        $<BUILD_INTERFACE:omnitrace::omnitrace-ptl>
        $<BUILD_INTERFACE:omnitrace::omnitrace-hip>
        $<BUILD_INTERFACE:omnitrace::omnitrace-roctracer>
        $<BUILD_INTERFACE:omnitrace::omnitrace-rocm-smi>
        $<BUILD_INTERFACE:$<IF:$<BOOL:${OMNITRACE_BUILD_LTO}>,omnitrace::omnitrace-lto,>>
        $<BUILD_INTERFACE:$<IF:$<BOOL:${OMNITRACE_BUILD_STATIC_LIBGCC}>,omnitrace::omnitrace-static-libgcc,>>
        $<BUILD_INTERFACE:$<IF:$<BOOL:${OMNITRACE_BUILD_STATIC_LIBSTDCXX}>,omnitrace::omnitrace-static-libstdcxx,>>
        $<BUILD_INTERFACE:$<IF:$<BOOL:${OMNITRACE_USE_SANITIZER}>,omnitrace::omnitrace-sanitizer,>>
    )

# ------------------------------------------------------------------------------#
#
# omnitrace object library
#
# ------------------------------------------------------------------------------#

add_library(omnitrace-object-library OBJECT)
add_library(omnitrace::omnitrace-object-library ALIAS omnitrace-object-library)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/library/defines.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/library/defines.hpp @ONLY)

set(library_sources
    ${CMAKE_CURRENT_LIST_DIR}/library.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/api.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/coverage.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/cpu_freq.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/critical_trace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/debug.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/dynamic_library.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/kokkosp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/mproc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/ompt.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/perfetto.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/process_sampler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/ptl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/runtime.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/sampling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/state.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/thread_data.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/timemory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/backtrace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/fork_gotcha.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/mpi_gotcha.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/omnitrace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/pthread_gotcha.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/pthread_create_gotcha.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/pthread_mutex_gotcha.cpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/user_region.cpp)

set(library_headers
    ${CMAKE_CURRENT_LIST_DIR}/library.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/api.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/config.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/common.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/coverage.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/cpu_freq.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/critical_trace.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/debug.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/dynamic_library.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/gpu.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/mproc.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/ompt.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/perfetto.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/process_sampler.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/ptl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/runtime.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/sampling.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/state.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/thread_data.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/timemory.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/utility.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/fwd.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/backtrace.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/fork_gotcha.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/functors.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/mpi_gotcha.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/omnitrace.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/rocm_smi.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/roctracer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/roctracer_callbacks.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/pthread_gotcha.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/pthread_create_gotcha.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/pthread_mutex_gotcha.hpp
    ${CMAKE_CURRENT_LIST_DIR}/library/components/user_region.hpp)

target_sources(omnitrace-object-library PRIVATE ${library_sources} ${library_headers})

if(OMNITRACE_USE_ROCTRACER)
    target_sources(
        omnitrace-object-library
        PRIVATE ${CMAKE_CURRENT_LIST_DIR}/library/components/roctracer.cpp
                ${CMAKE_CURRENT_LIST_DIR}/library/components/roctracer_callbacks.cpp)
endif()

if(OMNITRACE_USE_ROCM_SMI)
    target_sources(omnitrace-object-library
                   PRIVATE ${CMAKE_CURRENT_LIST_DIR}/library/components/rocm_smi.cpp)
endif()

target_link_libraries(omnitrace-object-library
                      PRIVATE omnitrace::omnitrace-interface-library)

if(OMNITRACE_DYNINST_API_RT)
    get_filename_component(OMNITRACE_DYNINST_API_RT_DIR "${OMNITRACE_DYNINST_API_RT}"
                           DIRECTORY)
endif()

# ------------------------------------------------------------------------------#
#
# omnitrace shared library
#
# ------------------------------------------------------------------------------#

add_library(omnitrace-library SHARED $<TARGET_OBJECTS:omnitrace-object-library>)
add_library(omnitrace::omnitrace-library ALIAS omnitrace-library)

target_link_libraries(omnitrace-library PRIVATE omnitrace::omnitrace-interface-library)

set_target_properties(
    omnitrace-library
    PROPERTIES OUTPUT_NAME omnitrace
               VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}
               INSTALL_RPATH "${OMNITRACE_LIB_INSTALL_RPATH}")

omnitrace_strip_target(omnitrace-library)

install(
    TARGETS omnitrace-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    OPTIONAL)
