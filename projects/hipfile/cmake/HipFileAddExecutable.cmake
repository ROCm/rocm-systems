# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(HipFileCompilerOptions)

# Add an executable program using hipFile build conventions
#
# Parameters:
#   NAME     <name>                             The name of the executable program to create
#   DEPS     [dependency1 [dependency2 ...]]    List of hipFile target library dependencies
#   SRCS     [src1 [src2 ...]]                  The source files
#   SYSINCLS [path1 [path2 ...]]                Paths to include dirs
#
# NOTE: Assumes hipFile target libraries are named <foo>_(static|shared)
#
# NOTE: This isn't the most robust function. It's mainly intended
#       to reduce code duplication. For example, DEPS is for passing
#       the hipFile shared/static dependency and won't work for passing
#       general library dependencies.
function(hipfile_add_executable)

    # Parse arguments
    set(options) # None at this time
    set(oneValueArgs NAME)
    set(multiValueArgs SRCS DEPS SYSINCLS)
    cmake_parse_arguments(arg "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_executable(${arg_NAME} ${arg_SRCS})
    hipfile_set_compiler_flags(${arg_NAME})

    # Set C++ standard
    target_compile_features(${arg_NAME} PRIVATE cxx_std_${HIPFILE_CXX_STANDARD})
    set_target_properties(${arg_NAME} PROPERTIES CXX_EXTENSIONS OFF)

    # Set position-independent code
    set_target_properties(${arg_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    get_target_property(linker_language ${arg_NAME} LINKER_LANGUAGE)
    if(linker_language STREQUAL "HIP")
        target_link_libraries(${arg_NAME} PRIVATE hip::device)
    endif()

    if(BUILD_TESTING)
        target_compile_definitions(${arg_NAME} PRIVATE AIS_TESTING)
    endif()

    if(CMAKE_HIP_PLATFORM STREQUAL "amd")
        target_compile_definitions(${arg_NAME} PRIVATE __HIP_PLATFORM_AMD__)
    elseif(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
        target_compile_definitions(${arg_NAME} PRIVATE __HIP_PLATFORM_NVIDIA__)
        target_include_directories(${arg_NAME} SYSTEM PRIVATE ${HIP_INCLUDE_DIRS})

        # CUDA include needs to come first. We get a CUB / Thrust version conflict otherwise.
        target_include_directories(${arg_NAME} SYSTEM BEFORE PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

        target_link_libraries(${arg_NAME} PRIVATE CUDA::cuda_driver)
        target_link_libraries(${arg_NAME} PRIVATE CUDA::cudart)
    endif()
    target_link_libraries(${arg_NAME} PRIVATE hip::host)

    foreach(dep IN LISTS arg_DEPS)
        add_dependencies(${arg_NAME} ${dep})
        target_link_libraries(${arg_NAME} PRIVATE ${dep})
    endforeach()

    foreach(incl IN LISTS arg_SYSINCLS)
        target_include_directories(${arg_NAME} SYSTEM PRIVATE ${incl})
    endforeach()

    target_include_directories(${arg_NAME} PRIVATE "${HIPFILE_ROOT_PATH}/shared")
endfunction()

# Add an executable test program using hipFile build conventions
#
# Parameters:
#   NAME     <name>                             The name of the executable program to create
#   DEPS     [dependency1 [dependency2 ...]]    List of hipFile target library dependencies
#   SRCS     [src1 [src2 ...]]                  The source files
#   SYSINCLS [path1 [path2 ...]]                Paths to include dirs
#
# NOTE: Simply a pass-through to add -UNDEBUG to test programs so they
#       always have assert() available.
function(hipfile_add_test_executable)

    # Parse arguments
    set(options) # None at this time
    set(oneValueArgs NAME)
    set(multiValueArgs SRCS DEPS SYSINCLS)
    cmake_parse_arguments(arg "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    hipfile_add_executable(
        NAME     ${arg_NAME}
        DEPS     ${arg_DEPS}
        SRCS     ${arg_SRCS}
        SYSINCLS ${arg_SYSINCLS}
    )
    target_compile_options(${arg_NAME} PRIVATE -UNDEBUG)

endfunction()
