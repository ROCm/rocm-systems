#
#
#
rocprofiler_activate_clang_tidy()

set(ROCPROFILER_LIB_HEADERS agent.hpp buffer.hpp external_correlation.hpp
                            intercept_table.hpp internal_threading.hpp registration.hpp)
set(ROCPROFILER_LIB_SOURCES
    agent.cpp
    buffer.cpp
    buffer_tracing.cpp
    device_counting_service.cpp
    callback_tracing.cpp
    context.cpp
    counters.cpp
    dispatch_profile.cpp
    external_correlation.cpp
    intercept_table.cpp
    internal_threading.cpp
    pc_sampling.cpp
    profile_config.cpp
    rocprofiler.cpp
    registration.cpp)

# ----------------------------------------------------------------------------------------#
#
# object library allows us to compile rocprofiler code once and generate both a shared and
# static library
#
# ----------------------------------------------------------------------------------------#

add_library(rocprofiler-object-library OBJECT)
add_library(rocprofiler-sdk::rocprofiler-object-library ALIAS rocprofiler-object-library)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_SOURCES}
                                                  ${ROCPROFILER_LIB_HEADERS})

add_subdirectory(hsa)
add_subdirectory(hip)
add_subdirectory(code_object)
add_subdirectory(context)
add_subdirectory(counters)
add_subdirectory(aql)
add_subdirectory(pc_sampling)
add_subdirectory(marker)
add_subdirectory(thread_trace)
add_subdirectory(tracing)
add_subdirectory(kernel_dispatch)
add_subdirectory(page_migration)
add_subdirectory(rccl)
add_subdirectory(details)

target_link_libraries(
    rocprofiler-object-library
    PUBLIC rocprofiler-sdk::rocprofiler-headers rocprofiler-sdk::rocprofiler-hip-nolink
           rocprofiler-sdk::rocprofiler-hsa-runtime-nolink
           rocprofiler-sdk::rocprofiler-rccl-nolink
    PRIVATE rocprofiler-sdk::rocprofiler-build-flags
            rocprofiler-sdk::rocprofiler-memcheck
            rocprofiler-sdk::rocprofiler-common-library
            rocprofiler-sdk::rocprofiler-amd-comgr
            rocprofiler-sdk::rocprofiler-hsa-aql
            rocprofiler-sdk::rocprofiler-drm
            rocprofiler-sdk::rocprofiler-hsakmt-nolink)

target_compile_definitions(rocprofiler-object-library PRIVATE rocprofiler_EXPORTS=1)

set_target_properties(rocprofiler-object-library PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ----------------------------------------------------------------------------------------#
#
# shared library
#
# ----------------------------------------------------------------------------------------#

add_library(rocprofiler-shared-library SHARED)
add_library(rocprofiler-sdk::rocprofiler-shared-library ALIAS rocprofiler-shared-library)

target_sources(
    rocprofiler-shared-library
    PRIVATE $<TARGET_OBJECTS:rocprofiler-sdk::rocprofiler-object-library>
            shared_library.cpp)
target_link_libraries(
    rocprofiler-shared-library
    INTERFACE rocprofiler-sdk::rocprofiler-headers
              $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-hsa-runtime-nolink>
              $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-hip-nolink>
    PRIVATE rocprofiler-sdk::rocprofiler-hsa-runtime-nolink
            rocprofiler-sdk::rocprofiler-hip-nolink
            rocprofiler-sdk::rocprofiler-build-flags
            rocprofiler-sdk::rocprofiler-memcheck
            rocprofiler-sdk::rocprofiler-common-library
            rocprofiler-sdk::rocprofiler-cxx-filesystem
            rocprofiler-sdk::rocprofiler-dl
            rocprofiler-sdk::rocprofiler-amd-comgr
            rocprofiler-sdk::rocprofiler-object-library)

set_target_properties(
    rocprofiler-shared-library
    PROPERTIES OUTPUT_NAME rocprofiler-sdk
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               DEFINE_SYMBOL rocprofiler_EXPORTS)

add_library(rocprofiler-sdk::rocprofiler-library ALIAS rocprofiler-shared-library)

install(
    TARGETS rocprofiler-shared-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT core
    EXPORT rocprofiler-sdk-targets)

# ----------------------------------------------------------------------------------------#
#
# static library (only built if another target links to it)
#
# ----------------------------------------------------------------------------------------#

add_library(rocprofiler-static-library EXCLUDE_FROM_ALL STATIC)
add_library(rocprofiler-sdk::rocprofiler-static-library ALIAS rocprofiler-static-library)

target_sources(rocprofiler-static-library
               PRIVATE $<TARGET_OBJECTS:rocprofiler-sdk::rocprofiler-object-library>)
target_link_libraries(
    rocprofiler-static-library
    PUBLIC rocprofiler-sdk::rocprofiler-headers
           rocprofiler-sdk::rocprofiler-hsa-runtime-nolink
           rocprofiler-sdk::rocprofiler-hip-nolink
    PRIVATE rocprofiler-sdk::rocprofiler-common-library
            rocprofiler-sdk::rocprofiler-object-library)

set_target_properties(
    rocprofiler-static-library PROPERTIES OUTPUT_NAME rocprofiler-sdk DEFINE_SYMBOL
                                                                      rocprofiler_EXPORTS)

# ----------------------------------------------------------------------------------------#
#
# testing
#
# ----------------------------------------------------------------------------------------#

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
