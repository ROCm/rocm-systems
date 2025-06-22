#
# C extensions are required
#
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------#
# target sources
# ------------------------------------------------------------------------------#

include(cmake/platform.cmake)

set(SRC_LIST src/RTcommon.c src/RTmemEmulator.c)

if(PLATFORM MATCHES freebsd)
    list(
        APPEND
        SRC_LIST
        src/RTposix.c
        src/RTfreebsd.c
        src/RTheap.c
        src/RTheap-freebsd.c
        src/RTthread.c
        src/RTspace.S
        src/RTsignal.c
    )
elseif(PLATFORM MATCHES linux)
    list(
        APPEND
        SRC_LIST
        src/RTposix.c
        src/RTlinux.c
        src/RTheap.c
        src/RTheap-linux.c
        src/RTthread.c
        src/RTspace.S
        src/RTsignal.c
    )
    list(
        APPEND
        RT_STATIC_ONLY_SRC_LIST
        src/RTstatic_ctors_dtors_begin.c
        src/RTstatic_ctors_dtors_end.c
    )
elseif(PLATFORM MATCHES nt OR PLATFORM MATCHES windows)
    list(
        APPEND
        SRC_LIST
        src/RTheap.c
        src/RTheap-win.c
        src/RTwinnt.c
        src/RTthread.c
        src/RTthread-x86.c
    )
endif()

set(SRC_LIST_i386 src/RTthread-x86.c src/RTtlsgetaddr-x86.S)
set(RT_STATIC_ONLY_SRC_LIST_i386 src/RTstatic_ctors_dtors-x86.c)
set(SRC_LIST_x86_64 src/RTthread-x86-64.c src/RTtlsgetaddr-x86.S)
set(RT_STATIC_ONLY_SRC_LIST_x86_64 src/RTstatic_ctors_dtors-x86.c)
set(SRC_LIST_ppc32 src/RTthread-powerpc.c src/RTthread-powerpc-asm.S)
set(RT_STATIC_ONLY_SRC_LIST_ppc32 src/RTstatic_ctors_dtors-ppc32.c)
set(SRC_LIST_ppc64 src/RTthread-powerpc.c src/RTthread-powerpc-asm.S)
set(RT_STATIC_ONLY_SRC_LIST_ppc64 src/RTstatic_ctors_dtors-ppc64.c)
set(SRC_LIST_aarch64
    src/RTthread-aarch64.c
    # src/RTthread-aarch64-asm.S
)
set(RT_STATIC_ONLY_SRC_LIST_aarch64 src/RTstatic_ctors_dtors-aarch64.c)

# We use gcc to compile the various assembly files, but cmake doesn't default to knowing
# that gcc can handle .S.
enable_language(ASM)
file(GLOB SRC_ASSEMBLY "src/*.S")
if(NEED_NATIVE_ASSEMBER)
    set_source_files_properties(${SRC_ASSEMBLY} PROPERTIES LANGUAGE ASM)
else()
    set_source_files_properties(${SRC_ASSEMBLY} PROPERTIES LANGUAGE C)
endif()

# The arch-specific files other than RTthread-x86 are Unix-only.
if(UNIX)
    if(PLATFORM MATCHES amd64 OR PLATFORM MATCHES x86_64)
        set(SRC_LIST_mabi ${SRC_LIST} ${SRC_LIST_i386})
        set(RT_STATIC_ONLY_SRC_LIST_mabi
            ${RT_STATIC_ONLY_SRC_LIST}
            ${RT_STATIC_ONLY_SRC_LIST_i386}
        )
        list(APPEND SRC_LIST ${SRC_LIST_x86_64})
        list(APPEND RT_STATIC_ONLY_SRC_LIST ${RT_STATIC_ONLY_SRC_LIST_x86_64})
    elseif(PLATFORM MATCHES ppc64)
        set(SRC_LIST_mabi ${SRC_LIST} ${SRC_LIST_ppc32})
        set(RT_STATIC_ONLY_SRC_LIST_mabi
            ${RT_STATIC_ONLY_SRC_LIST}
            ${RT_STATIC_ONLY_SRC_LIST_ppc32}
        )
        list(APPEND SRC_LIST ${SRC_LIST_ppc64})
        list(APPEND RT_STATIC_ONLY_SRC_LIST ${RT_STATIC_ONLY_SRC_LIST_ppc64})
    elseif(PLATFORM MATCHES i386)
        list(APPEND SRC_LIST ${SRC_LIST_i386})
        list(APPEND RT_STATIC_ONLY_SRC_LIST ${RT_STATIC_ONLY_SRC_LIST_i386})
    elseif(PLATFORM MATCHES ppc32)
        list(APPEND SRC_LIST ${SRC_LIST_ppc32})
        list(APPEND RT_STATIC_ONLY_SRC_LIST ${RT_STATIC_ONLY_SRC_LIST_ppc32})
    elseif(PLATFORM MATCHES aarch64)
        list(APPEND SRC_LIST ${SRC_LIST_aarch64})
        list(APPEND RT_STATIC_ONLY_SRC_LIST ${RT_STATIC_ONLY_SRC_LIST_aarch64})
    endif()
endif()

add_library(rocprofiler-systems-rt-library SHARED)
add_library(
    rocprofiler-systems::rocprofiler-systems-rt-library
    ALIAS rocprofiler-systems-rt-library
)

target_sources(rocprofiler-systems-rt-library PRIVATE ${SRC_LIST})
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
