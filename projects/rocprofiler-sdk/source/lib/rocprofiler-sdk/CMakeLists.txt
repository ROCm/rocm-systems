#
#
#
rocprofiler_activate_clang_tidy()

set(ROCPROFILER_LIB_HEADERS
    agent.hpp buffer.hpp external_correlation.hpp intercept_table.hpp
    internal_threading.hpp ompt.hpp registration.hpp runtime_initialization.hpp)
set(ROCPROFILER_LIB_SOURCES
    agent.cpp
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
add_subdirectory(rocdecode)
add_subdirectory(rocjpeg)
add_subdirectory(details)
add_subdirectory(ompt)

target_link_libraries(
    rocprofiler-sdk-object-library
    PUBLIC rocprofiler-sdk::rocprofiler-sdk-headers
           rocprofiler-sdk::rocprofiler-sdk-hip-nolink
           rocprofiler-sdk::rocprofiler-sdk-hsa-runtime-nolink
           rocprofiler-sdk::rocprofiler-sdk-rccl-nolink
           rocprofiler-sdk::rocprofiler-sdk-rocdecode-nolink
           rocprofiler-sdk::rocprofiler-sdk-rocjpeg-nolink
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-memcheck
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
            rocprofiler-sdk::rocprofiler-sdk-hsa-aql
            rocprofiler-sdk::rocprofiler-sdk-drm
            rocprofiler-sdk::rocprofiler-sdk-hsakmt-nolink
            rocprofiler-sdk::rocprofiler-sdk-dw)

target_compile_definitions(rocprofiler-sdk-object-library PRIVATE rocprofiler_EXPORTS=1)

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
    INTERFACE rocprofiler-sdk::rocprofiler-sdk-headers
              $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-hsa-runtime-nolink>
              $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-hip-nolink>
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-hsa-runtime-nolink
            rocprofiler-sdk::rocprofiler-sdk-hip-nolink
            rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-memcheck
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-cxx-filesystem
            rocprofiler-sdk::rocprofiler-sdk-dl
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
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
# static library (only built if another target links to it)
#
# ----------------------------------------------------------------------------------------#

add_library(rocprofiler-sdk-static-library EXCLUDE_FROM_ALL STATIC)
add_library(rocprofiler-sdk::rocprofiler-sdk-static-library ALIAS
            rocprofiler-sdk-static-library)

target_sources(rocprofiler-sdk-static-library
               PRIVATE $<TARGET_OBJECTS:rocprofiler-sdk::rocprofiler-sdk-object-library>)
target_link_libraries(
    rocprofiler-sdk-static-library
    PUBLIC rocprofiler-sdk::rocprofiler-sdk-headers
           rocprofiler-sdk::rocprofiler-sdk-hsa-runtime-nolink
           rocprofiler-sdk::rocprofiler-sdk-hip-nolink
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-dw>
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-object-library)

set_target_properties(
    rocprofiler-sdk-static-library PROPERTIES OUTPUT_NAME rocprofiler-sdk
                                              DEFINE_SYMBOL rocprofiler_EXPORTS)

# ----------------------------------------------------------------------------------------#
#
# testing
#
# ----------------------------------------------------------------------------------------#

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
