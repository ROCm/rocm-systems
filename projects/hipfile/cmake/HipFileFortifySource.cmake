# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# _FORTIFY_SOURCE flags for hipFile targets
#
# The language is a parameter, since hipFile compiles HIP sources as well
# as C++, and each language has its own CMAKE_<LANG>_FLAGS variables.

include_guard(GLOBAL)

# Check whether a string of compiler flags optimizes with -O2, -O3, or -Os
#
# The last -O option on the command line wins, so only that one is checked.
function(hipfile_flags_are_optimized outvar flags)
    separate_arguments(args UNIX_COMMAND "${flags}")
    set(last_opt_flag)
    foreach(arg IN LISTS args)
        if("${arg}" MATCHES "^-O")
            set(last_opt_flag "${arg}")
        endif()
    endforeach()

    if("${last_opt_flag}" MATCHES "^-O[23s]$")
        set(${outvar} TRUE PARENT_SCOPE)
    else()
        set(${outvar} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Get the flags that turn on _FORTIFY_SOURCE=3 for sources in ``language``
#
# _FORTIFY_SOURCE only works with optimization, so the flags are only
# used for configurations that build with -O2, -O3, or -Os. The flags
# are wrapped in a $<CONFIG:...> generator expression so this works
# with multi-config generators (Ninja Multi-Config, Xcode, etc.), where
# CMAKE_BUILD_TYPE is empty, as well as with single-config generators.
function(hipfile_get_fortify_flags outvar language)
    set(fortify_flags -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3)

    get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(is_multi_config)
        set(configs ${CMAKE_CONFIGURATION_TYPES})
    else()
        set(configs ${CMAKE_BUILD_TYPE})
    endif()

    # Without a build type, only the common flags are used
    if(NOT configs)
        hipfile_flags_are_optimized(optimized "${CMAKE_${language}_FLAGS}")
        if(optimized)
            set(${outvar} ${fortify_flags} PARENT_SCOPE)
        else()
            set(${outvar} "" PARENT_SCOPE)
        endif()
        return()
    endif()

    # CMake puts the per-configuration flags after the common flags, so
    # check them together (the per-configuration variables have upper-case
    # names, e.g. CMAKE_CXX_FLAGS_RELEASE)
    set(optimized_configs)
    foreach(config IN LISTS configs)
        string(TOUPPER "${config}" config_upper)
        hipfile_flags_are_optimized(optimized
            "${CMAKE_${language}_FLAGS} ${CMAKE_${language}_FLAGS_${config_upper}}")
        if(optimized)
            list(APPEND optimized_configs ${config})
        endif()
    endforeach()

    set(result)
    if(optimized_configs)
        list(JOIN optimized_configs "," config_list)
        foreach(flag IN LISTS fortify_flags)
            list(APPEND result "$<$<CONFIG:${config_list}>:${flag}>")
        endforeach()
    endif()

    set(${outvar} ${result} PARENT_SCOPE)
endfunction()
