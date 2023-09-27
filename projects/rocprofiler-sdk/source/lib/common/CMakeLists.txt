#
#   Builds common utilities into a static library
#
rocprofiler_activate_clang_tidy()

set(common_sources config.cpp environment.cpp demangle.cpp utility.cpp)
set(common_headers config.hpp defines.hpp environment.hpp demangle.hpp mpl.hpp
                   utility.hpp xml.hpp)

add_library(rocprofiler-common-library STATIC)
add_library(rocprofiler::rocprofiler-common-library ALIAS rocprofiler-common-library)

add_subdirectory(container)

target_sources(rocprofiler-common-library PRIVATE ${common_sources} ${common_headers})
target_include_directories(rocprofiler-common-library
                           PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)

target_link_libraries(
    rocprofiler-common-library
    PUBLIC $<BUILD_INTERFACE:rocprofiler::rocprofiler-headers>
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-build-flags>
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-threading>
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-memcheck>
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-stdcxxfs>
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-glog>
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-fmt>
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-dl>
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-hip>
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-amd-comgr>
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-hsa-runtime>
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-ptl>
           $<BUILD_INTERFACE:rocprofiler::rocprofiler-hsa-aql>)
set_target_properties(rocprofiler-common-library PROPERTIES OUTPUT_NAME
                                                            rocprofiler-common)
