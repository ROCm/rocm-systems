#
# C extensions are required
#
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Ensure .S files are compiled as assembly
set(NEED_NATIVE_ASSEMBLER ON)

# ------------------------------------------------------------------------------#
# target sources
# ------------------------------------------------------------------------------#

include(cmake/platform.cmake)

set(_public_headers h/dyninstAPI_RT.h h/dyninstRTExport.h)

set(_private_headers src/RTcommon.h src/RTheap.h src/RTthread.h)

set(_sources src/RTcommon.c src/RTheap.c src/RTthread.c)

if(PLATFORM MATCHES linux)
    set(_static_sources src/RTstatic_ctors_dtors_begin.c src/RTstatic_ctors_dtors_end.c)
    list(
        APPEND
        _sources
        src/RTposix.c
        src/RTlinux.c
        src/RTheap-linux.c
        src/RTspace.S
        src/RTsignal.c
    )
endif()

list(APPEND _sources src/RTthread-x86-64.c src/RTtlsgetaddr-x86.S)
list(APPEND _static_sources src/RTstatic_ctors_dtors-x86.c)

# We use gcc to compile the various assembly files, but cmake doesn't default to knowing
# that gcc can handle .S.
enable_language(ASM)
file(GLOB _source_assembly "src/*.S")

if(NEED_NATIVE_ASSEMBLER)
    set_source_files_properties(${_source_assembly} PROPERTIES LANGUAGE ASM)
else()
    set_source_files_properties(${_source_assembly} PROPERTIES LANGUAGE C)
endif()

add_library(rocprofiler-systems-rt-library SHARED)
add_library(
    rocprofiler-systems::rocprofiler-systems-rt-library
    ALIAS rocprofiler-systems-rt-library
)

target_sources(rocprofiler-systems-rt-library PRIVATE ${_sources})
target_include_directories(
    rocprofiler-systems-rt-library
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/h>
)
target_compile_definitions(rocprofiler-systems-rt-library PRIVATE ${UNIFIED_DEFINES})
target_link_libraries(
    rocprofiler-systems-rt-library
    PUBLIC $<BUILD_INTERFACE:${dl_LIBRARY}>
    PRIVATE rocprofiler-systems::rocprofiler-systems-threading
)

add_target_cxx_flag_if_avail(rocprofiler-systems-rt-library "-g3")

set_target_properties(
    rocprofiler-systems-rt-library
    PROPERTIES
        OUTPUT_NAME ${BINARY_NAME_PREFIX}-rt
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN"
)

rocprofiler_systems_strip_target(rocprofiler-systems-rt-library)

install(TARGETS rocprofiler-systems-rt-library DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(NOT EXISTS ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})
    file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})
endif()

add_custom_target(
    rocprofiler-systems-rt-library-dyninstAPI_RT-symlink
    ALL
    ${CMAKE_COMMAND} -E create_symlink
    ../$<TARGET_FILE_NAME:rocprofiler-systems-rt-library>
    ${CMAKE_SHARED_LIBRARY_PREFIX}dyninstAPI_RT${CMAKE_SHARED_LIBRARY_SUFFIX}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
    DEPENDS rocprofiler-systems-rt-library
    COMMENT
        "Creating ${CMAKE_SHARED_LIBRARY_PREFIX}dyninstAPI_RT${CMAKE_SHARED_LIBRARY_SUFFIX} to rocprof-sys-rt..."
)

install(
    FILES
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/${CMAKE_SHARED_LIBRARY_PREFIX}dyninstAPI_RT${CMAKE_SHARED_LIBRARY_SUFFIX}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
)
