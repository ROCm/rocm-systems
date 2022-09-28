# ------------------------------------------------------------------------------#
#
# omnitrace interface library
#
# ------------------------------------------------------------------------------#

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
    cmake_policy(SET CMP0115 NEW)
endif()

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
        $<BUILD_INTERFACE:omnitrace::omnitrace-bfd>
        $<BUILD_INTERFACE:omnitrace::omnitrace-mpi>
        $<BUILD_INTERFACE:omnitrace::omnitrace-ptl>
        $<BUILD_INTERFACE:omnitrace::omnitrace-hip>
        $<BUILD_INTERFACE:omnitrace::omnitrace-roctracer>
        $<BUILD_INTERFACE:omnitrace::omnitrace-rocprofiler>
        $<BUILD_INTERFACE:omnitrace::omnitrace-rocm-smi>
        $<BUILD_INTERFACE:omnitrace::omnitrace-rccl>
        $<BUILD_INTERFACE:omnitrace::omnitrace-static-libgcc-optional>
        $<BUILD_INTERFACE:omnitrace::omnitrace-static-libstdcxx-optional>
        $<BUILD_INTERFACE:omnitrace::omnitrace-sanitizer>
        $<BUILD_INTERFACE:$<IF:$<BOOL:${OMNITRACE_BUILD_LTO}>,omnitrace::omnitrace-lto,>>)

# ------------------------------------------------------------------------------#
#
# omnitrace object library
#
# ------------------------------------------------------------------------------#

add_library(omnitrace-object-library OBJECT)
add_library(omnitrace::omnitrace-object-library ALIAS omnitrace-object-library)

target_sources(
    omnitrace-object-library
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/library.cpp ${CMAKE_CURRENT_LIST_DIR}/api.cpp
            ${CMAKE_CURRENT_LIST_DIR}/api.hpp)

add_subdirectory(library)

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
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               INSTALL_RPATH "${OMNITRACE_LIB_INSTALL_RPATH}")

omnitrace_strip_target(omnitrace-library)

install(
    TARGETS omnitrace-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    OPTIONAL)
