#
#
#
rocprofiler_activate_clang_tidy()

if(WIN32)
    # Windows builds compile only the core subset that does not require Linux-specific
    # headers (ELF/DWARF, DRM, KFD ioctl, dlfcn, pthreads) or complex C++ patterns
    # (std::integer_sequence<enum>) that MSVC rejects or ICEs on.
    set(ROCPROFILER_LIB_HEADERS agent.hpp buffer.hpp external_correlation.hpp
                                internal_threading.hpp registration.hpp)
    set(ROCPROFILER_LIB_SOURCES
        agent.cpp
        agent_windows.cpp
        buffer.cpp
        buffer_tracing.cpp
        context.cpp
        counter_config.cpp
        counters.cpp
        dispatch_counting_service.cpp
        external_correlation.cpp
        internal_threading.cpp
        registration.cpp
        rocprofiler.cpp)
else()
    set(ROCPROFILER_LIB_HEADERS
        agent.hpp buffer.hpp external_correlation.hpp intercept_table.hpp
        internal_threading.hpp ompt.hpp registration.hpp runtime_initialization.hpp)
    set(ROCPROFILER_LIB_SOURCES
        agent.cpp
        attach.cpp
        buffer.cpp
        buffer_tracing.cpp
        callback_tracing.cpp
        context.cpp
        counter_config.cpp
        counters.cpp
        device_counting_service.cpp
        dispatch_counting_service.cpp
        external_correlation.cpp
        intercept_table.cpp
        internal_threading.cpp
        ompt.cpp
        pc_sampling.cpp
        registration.cpp
        rocprofiler.cpp
        runtime_initialization.cpp)
endif()

# ----------------------------------------------------------------------------------------#
#
# object library allows us to compile rocprofiler code once and generate both a shared and
# static library
#
# ----------------------------------------------------------------------------------------#

add_library(rocprofiler-sdk-object-library OBJECT)
add_library(rocprofiler-sdk::rocprofiler-sdk-object-library ALIAS
            rocprofiler-sdk-object-library)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_SOURCES}
                                                      ${ROCPROFILER_LIB_HEADERS})

add_subdirectory(counters)
add_subdirectory(context)
add_subdirectory(platform)
add_subdirectory(hsa)

add_subdirectory(hip)
add_subdirectory(tracing)

# Subsystems reached from code that does build on Windows. Each one's CMakeLists selects a
# windows_stubs.cpp in place of its Linux sources, so the call sites in the shared code
# stay identical across platforms and the stub file is deleted wholesale when the real
# port lands.
add_subdirectory(aql)
add_subdirectory(kernel_replay)
add_subdirectory(registration)
add_subdirectory(spm)
add_subdirectory(thread_trace)

# Subsystems that are still Linux-only: they depend on ELF/DWARF, ptrace, dlfcn, or
# runtime API tables that have no Windows equivalent yet.
if(NOT WIN32)
    add_subdirectory(kernel_dispatch)
    add_subdirectory(details)
    add_subdirectory(hipfile)
    add_subdirectory(code_object)
    add_subdirectory(marker)
    add_subdirectory(rccl)
    add_subdirectory(rocdecode)
    add_subdirectory(rocjpeg)
    add_subdirectory(rocshmem)
    add_subdirectory(ompt)
endif()

if(ROCPROFILER_BUILD_KFD OR WIN32)
    add_subdirectory(kfd)
endif()

if(ROCPROFILER_BUILD_PC_SAMPLING OR WIN32)
    add_subdirectory(pc_sampling)
endif()

target_link_libraries(
    rocprofiler-sdk-object-library
    PUBLIC rocprofiler-sdk::rocprofiler-sdk-headers
    PRIVATE
        rocprofiler-sdk::rocprofiler-sdk-build-flags
        rocprofiler-sdk::rocprofiler-sdk-sanitizer
        rocprofiler-sdk::rocprofiler-sdk-common-library
        rocprofiler-sdk::rocprofiler-sdk-amd-comgr
        $<$<BOOL:${WIN32}>:rocprofiler-sdk::rocprofiler-sdk-etw>
        $<$<NOT:$<BOOL:${WIN32}>>:rocprofiler-sdk::rocprofiler-sdk-aqlprofile>
        $<$<NOT:$<BOOL:${WIN32}>>:rocprofiler-sdk::rocprofiler-sdk-drm>
        $<$<NOT:$<BOOL:${WIN32}>>:rocprofiler-sdk::rocprofiler-sdk-dw>
        $<$<NOT:$<BOOL:${WIN32}>>:rocprofiler-sdk::rocprofiler-sdk-rocprof-trace-decoder>)

target_compile_definitions(
    rocprofiler-sdk-object-library
    PRIVATE rocprofiler_EXPORTS=1
            $<$<BOOL:${ROCPROFILER_BUILD_AQLPROFILE}>:ROCPROFILER_BUILD_AQLPROFILE=1>
            $<$<BOOL:${ROCPROFILER_BUILD_KFD}>:ROCPROFILER_BUILD_KFD=1>
            $<$<BOOL:${ROCPROFILER_BUILD_PC_SAMPLING}>:ROCPROFILER_BUILD_PC_SAMPLING=1>)

set_target_properties(rocprofiler-sdk-object-library PROPERTIES POSITION_INDEPENDENT_CODE
                                                                ON)

# ----------------------------------------------------------------------------------------#
#
# shared library
#
# ----------------------------------------------------------------------------------------#

add_library(rocprofiler-sdk-shared-library SHARED)
add_library(rocprofiler-sdk::rocprofiler-sdk-shared-library ALIAS
            rocprofiler-sdk-shared-library)

target_sources(
    rocprofiler-sdk-shared-library
    PRIVATE $<TARGET_OBJECTS:rocprofiler-sdk::rocprofiler-sdk-object-library>
            shared_library.cpp)
target_link_libraries(
    rocprofiler-sdk-shared-library
    PUBLIC rocprofiler-sdk::rocprofiler-sdk-headers
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-memcheck
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-cxx-filesystem
            rocprofiler-sdk::rocprofiler-sdk-dl
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
            $<$<BOOL:${WIN32}>:rocprofiler-sdk::rocprofiler-sdk-etw>
            rocprofiler-sdk::rocprofiler-sdk-object-library)

set_target_properties(
    rocprofiler-sdk-shared-library
    PROPERTIES OUTPUT_NAME rocprofiler-sdk
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               DEFINE_SYMBOL rocprofiler_EXPORTS)

add_library(rocprofiler-sdk::rocprofiler-library ALIAS rocprofiler-sdk-shared-library)

install(
    TARGETS rocprofiler-sdk-shared-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT core
    EXPORT rocprofiler-sdk-targets)

# ----------------------------------------------------------------------------------------#
#
# unversioned shared library used to test loading two instances of rocprofiler-sdk
#
# ----------------------------------------------------------------------------------------#

if(ROCPROFILER_BUILD_TESTS)
    add_library(rocprofiler-sdk-test-duplicate-shared-library SHARED)

    target_sources(
        rocprofiler-sdk-test-duplicate-shared-library
        PRIVATE $<TARGET_OBJECTS:rocprofiler-sdk::rocprofiler-sdk-object-library>
                shared_library.cpp)
    target_link_libraries(
        rocprofiler-sdk-test-duplicate-shared-library
        PUBLIC rocprofiler-sdk::rocprofiler-sdk-headers
        PRIVATE rocprofiler-sdk::rocprofiler-sdk-build-flags
                rocprofiler-sdk::rocprofiler-sdk-memcheck
                rocprofiler-sdk::rocprofiler-sdk-common-library
                rocprofiler-sdk::rocprofiler-sdk-cxx-filesystem
                rocprofiler-sdk::rocprofiler-sdk-dl
                rocprofiler-sdk::rocprofiler-sdk-amd-comgr
                rocprofiler-sdk::rocprofiler-sdk-object-library)

    set_target_properties(
        rocprofiler-sdk-test-duplicate-shared-library
        PROPERTIES OUTPUT_NAME rocprofiler-sdk
                   LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests/duplicate-sdk"
                   BUILD_RPATH "\$ORIGIN"
                   DEFINE_SYMBOL rocprofiler_EXPORTS)

    install(
        TARGETS rocprofiler-sdk-test-duplicate-shared-library
        LIBRARY
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/duplicate-sdk
            COMPONENT tests)
endif()

# ----------------------------------------------------------------------------------------#
#
# static library (only built if another target links to it)
#
# ----------------------------------------------------------------------------------------#

add_library(rocprofiler-sdk-static-library EXCLUDE_FROM_ALL STATIC)
add_library(rocprofiler-sdk::rocprofiler-sdk-static-library ALIAS
            rocprofiler-sdk-static-library)

target_sources(rocprofiler-sdk-static-library
               PRIVATE $<TARGET_OBJECTS:rocprofiler-sdk::rocprofiler-sdk-object-library>)
if(WIN32)
    target_link_libraries(
        rocprofiler-sdk-static-library
        PUBLIC rocprofiler-sdk::rocprofiler-sdk-headers
        PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
                rocprofiler-sdk::rocprofiler-sdk-object-library)
else()
    target_link_libraries(
        rocprofiler-sdk-static-library
        PUBLIC rocprofiler-sdk::rocprofiler-sdk-headers
               $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-dw>
               $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-aqlprofile>
        PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
                rocprofiler-sdk::rocprofiler-sdk-object-library)
endif()

if(WIN32)
    # On Windows, the DLL's import lib is also named rocprofiler-sdk.lib — use a distinct
    # name to avoid the Ninja "multiple rules generate" collision.
    set_target_properties(
        rocprofiler-sdk-static-library PROPERTIES OUTPUT_NAME rocprofiler-sdk-static
                                                  DEFINE_SYMBOL rocprofiler_EXPORTS)
else()
    set_target_properties(
        rocprofiler-sdk-static-library PROPERTIES OUTPUT_NAME rocprofiler-sdk
                                                  DEFINE_SYMBOL rocprofiler_EXPORTS)
endif()

# ----------------------------------------------------------------------------------------#
#
# testing
#
# ----------------------------------------------------------------------------------------#

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(NOT WIN32)
    add_subdirectory(rocprofiler-avail)
endif()
