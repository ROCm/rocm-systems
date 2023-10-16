#
#
#
rocprofiler_activate_clang_tidy()

set(ROCPROFILER_LIB_HEADERS buffer.hpp internal_threading.hpp registration.hpp)
set(ROCPROFILER_LIB_SOURCES
    agent.cpp
    buffer.cpp
    buffer_tracing.cpp
    callback_tracing.cpp
    context.cpp
    counters.cpp
    dispatch_profile.cpp
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
add_library(rocprofiler::rocprofiler-object-library ALIAS rocprofiler-object-library)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_SOURCES}
                                                  ${ROCPROFILER_LIB_HEADERS})

add_subdirectory(hsa)
add_subdirectory(context)
add_subdirectory(counters)
add_subdirectory(aql)

target_link_libraries(
    rocprofiler-object-library
    PUBLIC rocprofiler::rocprofiler-headers rocprofiler::rocprofiler-hsa-runtime
           rocprofiler::rocprofiler-hip
    PRIVATE rocprofiler::rocprofiler-build-flags rocprofiler::rocprofiler-memcheck
            rocprofiler::rocprofiler-common-library rocprofiler::rocprofiler-stdcxxfs
            rocprofiler::rocprofiler-dl rocprofiler::rocprofiler-amd-comgr)

target_compile_definitions(rocprofiler-object-library PRIVATE rocprofiler_EXPORTS=1)

set_target_properties(rocprofiler-object-library PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ----------------------------------------------------------------------------------------#
#
# shared library
#
# ----------------------------------------------------------------------------------------#

add_library(rocprofiler-shared-library SHARED)
add_library(rocprofiler::rocprofiler-shared-library ALIAS rocprofiler-shared-library)

target_sources(rocprofiler-shared-library
               PRIVATE $<TARGET_OBJECTS:rocprofiler::rocprofiler-object-library>)
target_link_libraries(
    rocprofiler-shared-library
    INTERFACE rocprofiler::rocprofiler-headers rocprofiler::rocprofiler-hsa-runtime
              rocprofiler::rocprofiler-hip
    PRIVATE rocprofiler::rocprofiler-build-flags
            rocprofiler::rocprofiler-memcheck
            rocprofiler::rocprofiler-common-library
            rocprofiler::rocprofiler-stdcxxfs
            rocprofiler::rocprofiler-dl
            rocprofiler::rocprofiler-amd-comgr
            rocprofiler::rocprofiler-object-library)

set_target_properties(
    rocprofiler-shared-library
    PROPERTIES OUTPUT_NAME rocprofiler64
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               SKIP_BUILD_RPATH OFF
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               DEFINE_SYMBOL rocprofiler_EXPORTS)

add_library(rocprofiler::rocprofiler-library ALIAS rocprofiler-shared-library)
add_library(rocprofiler::rocprofiler ALIAS rocprofiler-shared-library)

install(
    TARGETS rocprofiler-shared-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT core
    EXPORT ${PROJECT_NAME}-library-targets)

# ----------------------------------------------------------------------------------------#
#
# static library (only built if another target links to it)
#
# ----------------------------------------------------------------------------------------#

add_library(rocprofiler-static-library EXCLUDE_FROM_ALL STATIC)
add_library(rocprofiler::rocprofiler-static-library ALIAS rocprofiler-static-library)

target_sources(rocprofiler-static-library
               PRIVATE $<TARGET_OBJECTS:rocprofiler::rocprofiler-object-library>)
target_link_libraries(
    rocprofiler-static-library
    PUBLIC rocprofiler::rocprofiler-headers rocprofiler::rocprofiler-hsa-runtime
           rocprofiler::rocprofiler-hip
    PRIVATE rocprofiler::rocprofiler-common-library
            rocprofiler::rocprofiler-object-library)

set_target_properties(
    rocprofiler-static-library PROPERTIES OUTPUT_NAME rocprofiler64 DEFINE_SYMBOL
                                                                    rocprofiler_EXPORTS)

# ----------------------------------------------------------------------------------------#
#
# testing
#
# ----------------------------------------------------------------------------------------#

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
