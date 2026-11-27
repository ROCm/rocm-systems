# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Configure-only test for cmake/HipFileFortifySource.cmake
#
# The hipfile_fortify_source tests in test/CMakeLists.txt configure this
# project with a single-config generator and, when one is available, a
# multi-config generator. Each case sets the build type and flag variables
# that hipfile_get_fortify_flags() reads, in a block() so they don't leak
# into the next case, and checks the flags it returns.

cmake_minimum_required(VERSION 3.25.2)

project(hipfile_fortify_source_test LANGUAGES NONE)

include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/HipFileFortifySource.cmake")

set(fortify_flags -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3)

# Fail unless hipfile_get_fortify_flags(... ${language}) returns ``expected``
function(check_fortify_flags name language expected)
    hipfile_get_fortify_flags(actual ${language})
    if(NOT "${actual}" STREQUAL "${expected}")
        message(FATAL_ERROR "${name}: got '${actual}', expected '${expected}'")
    endif()
endfunction()

# The fortify flags, enabled only for the configurations in ``configs``
# (a comma-separated list)
function(get_config_flags outvar configs)
    set(result)
    foreach(flag IN LISTS fortify_flags)
        list(APPEND result "$<$<CONFIG:${configs}>:${flag}>")
    endforeach()
    set(${outvar} ${result} PARENT_SCOPE)
endfunction()

get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

if(is_multi_config)
    # Only the optimized configurations are fortified, and CMAKE_BUILD_TYPE
    # is ignored
    block()
        set(CMAKE_CONFIGURATION_TYPES Debug Release RelWithDebInfo MinSizeRel)
        set(CMAKE_BUILD_TYPE Debug)
        set(CMAKE_CXX_FLAGS "")
        set(CMAKE_CXX_FLAGS_DEBUG "-g")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
        set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
        get_config_flags(expected "Release,RelWithDebInfo,MinSizeRel")
        check_fortify_flags("multi-config" CXX "${expected}")
    endblock()

    # No optimized configuration
    block()
        set(CMAKE_CONFIGURATION_TYPES Debug)
        set(CMAKE_CXX_FLAGS "")
        set(CMAKE_CXX_FLAGS_DEBUG "-g")
        check_fortify_flags("multi-config Debug only" CXX "")
    endblock()

    return()
endif()

# Each standard optimized build type, including the RelWithDebInfo default,
# whose flags are in CMAKE_<LANG>_FLAGS_RELWITHDEBINFO
foreach(case IN ITEMS "Release|-O3 -DNDEBUG" "RelWithDebInfo|-O2 -g -DNDEBUG" "MinSizeRel|-Os -DNDEBUG")
    string(REPLACE "|" ";" case "${case}")
    list(GET case 0 build_type)
    list(GET case 1 build_type_flags)
    block()
        set(CMAKE_BUILD_TYPE ${build_type})
        string(TOUPPER "${build_type}" build_type_upper)
        set(CMAKE_CXX_FLAGS "")
        set(CMAKE_CXX_FLAGS_${build_type_upper} "${build_type_flags}")
        get_config_flags(expected "${build_type}")
        check_fortify_flags("${build_type}" CXX "${expected}")
    endblock()
endforeach()

# Unoptimized build types
block()
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS "")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    check_fortify_flags("Debug" CXX "")
endblock()

block()
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "")
    foreach(opt_flag IN ITEMS -O0 -O1 -Og -Ofast)
        set(CMAKE_CXX_FLAGS_RELEASE "${opt_flag}")
        check_fortify_flags("${opt_flag}" CXX "")
    endforeach()
endblock()

# The build type's case doesn't matter
block()
    set(CMAKE_BUILD_TYPE release)
    set(CMAKE_CXX_FLAGS "")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    get_config_flags(expected "release")
    check_fortify_flags("lower-case build type" CXX "${expected}")
endblock()

# The last -O option wins, and the per-configuration flags come last
block()
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS "-O2")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    check_fortify_flags("-O2 then -O0" CXX "")
endblock()

block()
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "-O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    get_config_flags(expected "Release")
    check_fortify_flags("-O0 then -O3" CXX "${expected}")
endblock()

block()
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -O1")
    check_fortify_flags("-O3 then -O1" CXX "")
endblock()

# Without a build type, only the common flags count, and no generator
# expression is needed
block()
    set(CMAKE_BUILD_TYPE "")
    set(CMAKE_CXX_FLAGS "-O2")
    set(CMAKE_CXX_FLAGS_RELEASE "-O0")
    check_fortify_flags("no build type, -O2" CXX "${fortify_flags}")
    set(CMAKE_CXX_FLAGS "")
    check_fortify_flags("no build type, no flags" CXX "")
endblock()

# Each language uses its own flags
block()
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_HIP_FLAGS "")
    set(CMAKE_HIP_FLAGS_RELEASE "-O0")
    get_config_flags(expected "Release")
    check_fortify_flags("CXX -O3" CXX "${expected}")
    check_fortify_flags("HIP -O0" HIP "")
endblock()
