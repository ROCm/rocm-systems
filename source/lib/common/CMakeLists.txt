#
#   Builds common utilities into a static library
#
rocprofiler_activate_clang_tidy()

set(common_sources
    demangle.cpp
    elf_utils.cpp
    environment.cpp
    logging.cpp
    md5sum.cpp
    simple_timer.cpp
    static_object.cpp
    static_tl_object.cpp
    string_entry.cpp
    utility.cpp)
set(common_headers
    abi.hpp
    defines.hpp
    demangle.hpp
    elf_utils.hpp
    environment.hpp
    filesystem.hpp
    hasher.hpp
    logging.hpp
    md5sum.hpp
    mpl.hpp
    scope_destructor.hpp
    simple_timer.hpp
    static_object.hpp
    static_tl_object.hpp
    string_entry.hpp
    stringize_arg.hpp
    synchronized.hpp
    units.hpp
    utility.hpp)

add_library(rocprofiler-sdk-common-library STATIC)
add_library(rocprofiler-sdk::rocprofiler-sdk-common-library ALIAS
            rocprofiler-sdk-common-library)

add_subdirectory(container)
add_subdirectory(memory)

target_sources(rocprofiler-sdk-common-library PRIVATE ${common_sources} ${common_headers})
target_include_directories(rocprofiler-sdk-common-library
                           PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)

target_link_libraries(
    rocprofiler-sdk-common-library
    PUBLIC $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-headers>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-build-flags>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-threading>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-memcheck>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-cxx-filesystem>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-glog>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-fmt>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-yaml-cpp>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-dl>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-ptl>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-atomic>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-hsakmt-nolink>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-elfio>)

set_target_properties(rocprofiler-sdk-common-library PROPERTIES OUTPUT_NAME
                                                                rocprofiler-sdk-common)

add_subdirectory(details)
