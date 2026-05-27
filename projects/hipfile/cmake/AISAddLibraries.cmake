# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(AISCompilerOptions)
include(CheckLinkerFlag)

# Add a library using AIS build conventions. Shared vs. static
# is decided by the BUILD_SHARED_LIBS option and only one can
# be built at a time (like the rest of ROCm).
#
# Parameters:
#   NAME     <name>                             The name of the libraries to create
#   DETAIL   "amd"|"nvidia"                     Whether this is an AMD or NVIDIA target
#   LIBINCL  <path>                             Path to installed include files (e.g., hipfile.h)
#   SYSINCLS [path1 [path2 ...]]                Paths to other include dirs
#   SRCS     [src1 [src2 ...]]                  The source files
#   LIBS     [lib1, [lib2 ...]]                 List of external library dependencies
function(ais_add_libraries)

    # Parse arguments
    set(options) # None at this time
    set(oneValueArgs NAME DETAIL LIBINCL)
    set(multiValueArgs SRCS SYSINCLS LIBS)
    cmake_parse_arguments(arg "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Add the library target
    add_library(${arg_NAME} ${arg_SRCS})

    # Set C++ standard
    target_compile_features(${arg_NAME} PUBLIC cxx_std_${AIS_CXX_STANDARD})
    set_target_properties(${arg_NAME} PROPERTIES CXX_EXTENSIONS OFF)

    # Set position-independent code
    set_target_properties(${arg_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # Add version numbers
    set_target_properties(${arg_NAME} PROPERTIES VERSION ${AIS_LIBRARY_VERSION})
    if(BUILD_SHARED_LIBS)
        set_target_properties(${arg_NAME} PROPERTIES SOVERSION ${AIS_LIBRARY_SOVERSION})
    endif()

    # Add dependencies on external libraries
    foreach(lib IN LISTS arg_LIBS)
        find_library(LIBRARY_PATH NAMES ${lib})
        if(NOT LIBRARY_PATH)
            message(FATAL_ERROR "lib${lib} not found")
        endif()
        target_link_libraries(${arg_NAME} PRIVATE ${LIBRARY_PATH})
    endforeach()

    if(BUILD_TESTING)
        target_compile_definitions(${arg_NAME} PRIVATE AIS_TESTING)
    endif()

    if(arg_DETAIL STREQUAL "nvidia")
        target_compile_definitions(${arg_NAME} PRIVATE __HIP_PLATFORM_NVIDIA__)
    else()
        target_compile_definitions(${arg_NAME} PRIVATE __HIP_PLATFORM_AMD__)
    endif()

    # Set the library name, minus the .a, .so, or whatever
    set_target_properties(${arg_NAME} PROPERTIES OUTPUT_NAME ${arg_NAME})

    # Set the public include paths (these are installed libraries)
    target_include_directories(${arg_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${arg_LIBINCL}>
        $<INSTALL_INTERFACE:include>
    )

    # Add other include paths
    foreach(incl IN LISTS arg_SYSINCLS)
        target_include_directories(${arg_NAME} SYSTEM PRIVATE ${incl})
    endforeach()

    # Add libraries and headers for NVIDIA targets
    if(arg_DETAIL STREQUAL "nvidia")
        target_include_directories(${arg_NAME}
            SYSTEM
            PRIVATE
            ${HIP_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_link_libraries(${arg_NAME} PRIVATE ${CUFILE_LIBRARY})
    endif()
    target_link_libraries(${arg_NAME} PRIVATE hip::host)

    # Add link options (mainly for hardening)
    check_linker_flag(CXX "-Wl,-z,noexecstack" LINKER_SUPPORTS_NOEXECSTACK)
    if(LINKER_SUPPORTS_NOEXECSTACK)
        target_link_options(${arg_NAME} PRIVATE "-Wl,-z,noexecstack")
    endif()

    # Add the common include path
    target_include_directories(${arg_NAME} PRIVATE "${HIPFILE_ROOT_PATH}/shared")

    # Set compiler flags
    ais_set_compiler_flags(${arg_NAME})
endfunction()
