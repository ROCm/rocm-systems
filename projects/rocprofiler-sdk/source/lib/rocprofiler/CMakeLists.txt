set(ROCPROFILER_LIB_HEADERS config_helpers.hpp config_internal.hpp tracer.hpp)
set(ROCPROFILER_LIB_SOURCES config_internal.cpp rocprofiler_config.cpp)

add_library(rocprofiler-library SHARED)
add_library(rocprofiler::rocprofiler-library ALIAS rocprofiler-library)

target_sources(rocprofiler-library PRIVATE ${ROCPROFILER_LIB_SOURCES}
                                           ${ROCPROFILER_LIB_HEADERS})

add_subdirectory(hsa)

target_link_libraries(
    rocprofiler-library
    PUBLIC rocprofiler::rocprofiler-headers
    PRIVATE rocprofiler::rocprofiler-build-flags
            rocprofiler::rocprofiler-memcheck
            rocprofiler::rocprofiler-common-library
            rocprofiler::rocprofiler-stdcxxfs
            rocprofiler::rocprofiler-dl
            rocprofiler::rocprofiler-hip
            rocprofiler::rocprofiler-amd-comgr
            rocprofiler::rocprofiler-hsa-runtime)
target_compile_definitions(
    rocprofiler-library PRIVATE AMD_INTERNAL_BUILD=1 PROF_API_IMPL=1
                                HIP_PROF_HIP_API_STRING=1 __HIP_PLATFORM_AMD__=1)
set_target_properties(
    rocprofiler-library
    PROPERTIES OUTPUT_NAME rocprofiler64
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION})
