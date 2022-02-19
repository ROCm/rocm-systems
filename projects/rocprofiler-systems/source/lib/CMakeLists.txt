# ------------------------------------------------------------------------------#
#
# omnitrace interface library
#
# ------------------------------------------------------------------------------#

add_library(omnitrace-interface-library INTERFACE)
add_library(omnitrace::omnitrace-interface-library ALIAS omnitrace-interface-library)

target_include_directories(
    omnitrace-interface-library INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                          ${CMAKE_CURRENT_BINARY_DIR}/include)
target_include_directories(omnitrace-interface-library SYSTEM
                           INTERFACE ${perfetto_DIR}/sdk)

target_compile_definitions(
    omnitrace-interface-library
    INTERFACE OMNITRACE_MAX_THREADS=${OMNITRACE_MAX_THREADS}
              $<IF:$<BOOL:${OMNITRACE_CUSTOM_DATA_SOURCE}>,CUSTOM_DATA_SOURCE,>)

target_link_libraries(
    omnitrace-interface-library
    INTERFACE $<BUILD_INTERFACE:omnitrace::omnitrace-headers>
              $<BUILD_INTERFACE:omnitrace::omnitrace-threading>
              $<BUILD_INTERFACE:omnitrace::omnitrace-compile-options>
              $<BUILD_INTERFACE:omnitrace::omnitrace-hip>
              $<BUILD_INTERFACE:omnitrace::omnitrace-roctracer>
              $<BUILD_INTERFACE:omnitrace::omnitrace-rocm-smi>
              $<BUILD_INTERFACE:omnitrace::omnitrace-mpi>
              $<BUILD_INTERFACE:omnitrace::omnitrace-ptl>
              $<BUILD_INTERFACE:timemory::timemory-headers>
              $<BUILD_INTERFACE:timemory::timemory-gotcha>
              $<BUILD_INTERFACE:timemory::timemory-cxx-shared>
              $<IF:$<BOOL:${OMNITRACE_USE_SANITIZER}>,omnitrace::omnitrace-sanitizer,>)

# ------------------------------------------------------------------------------#
#
# omnitrace object library
#
# ------------------------------------------------------------------------------#

add_library(omnitrace-object-library OBJECT)
add_library(omnitrace::omnitrace-object-library ALIAS omnitrace-object-library)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/library/defines.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/library/defines.hpp @ONLY)

set(library_sources
    ${CMAKE_CURRENT_LIST_DIR}/src/library.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/cpu_freq.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/critical_trace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/kokkosp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/perfetto.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/ptl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/sampling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/state.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/thread_data.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/thread_sampler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/timemory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/components/backtrace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/components/fork_gotcha.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/components/mpi_gotcha.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/components/omnitrace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/components/pthread_gotcha.cpp
    ${perfetto_DIR}/sdk/perfetto.cc)

set(library_headers
    ${CMAKE_CURRENT_LIST_DIR}/include/library.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/api.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/config.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/common.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/cpu_freq.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/critical_trace.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/debug.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/gpu.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/perfetto.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/ptl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/sampling.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/state.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/thread_data.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/thread_sampler.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/timemory.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/components/fwd.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/components/backtrace.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/components/fork_gotcha.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/components/mpi_gotcha.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/components/omnitrace.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/components/rocm_smi.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/components/roctracer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/components/roctracer_callbacks.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/components/pthread_gotcha.hpp
    ${perfetto_DIR}/sdk/perfetto.h)

target_sources(omnitrace-object-library PRIVATE ${library_sources} ${library_headers})

if(OMNITRACE_USE_ROCTRACER)
    target_sources(
        omnitrace-object-library
        PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/library/components/roctracer.cpp
                ${CMAKE_CURRENT_LIST_DIR}/src/library/components/roctracer_callbacks.cpp)
endif()

if(OMNITRACE_USE_ROCM_SMI)
    target_sources(omnitrace-object-library
                   PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/library/components/rocm_smi.cpp)
endif()

target_link_libraries(omnitrace-object-library PRIVATE omnitrace-interface-library)

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

target_link_libraries(omnitrace-library PRIVATE omnitrace-interface-library)

set_target_properties(
    omnitrace-library
    PROPERTIES OUTPUT_NAME omnitrace
               VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}
               INSTALL_RPATH
               "\$ORIGIN:\$ORIGIN/timemory/libunwind:\$ORIGIN/dyninst-tpls/libs")

install(
    TARGETS omnitrace-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    OPTIONAL)
