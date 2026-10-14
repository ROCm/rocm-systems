# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

#-----------------------------------------------------------------------------
# Backward compatibility for the deprecated AIS_* CMake options
#
# The public hipFile CMake options were renamed from AIS_<name> to
# HIPFILE_<name>. This module keeps the old names working while downstream
# projects (e.g., TheRock) and build scripts switch over.
#
# For each deprecated option that is defined, either on the command line or in
# an existing CMakeCache.txt, the value is copied to the new option and the old
# cache entry is removed. Since the old entry is removed on every configure, a
# defined AIS_<name> is always a fresh request, so it overrides any
# HIPFILE_<name> value that is already in the cache.
#
# The new cache entry is given the UNINITIALIZED type, which makes it look
# exactly like -DHIPFILE_<name>=<value> on the command line. The option() or
# set(... CACHE ...) that declares the option later fills in the real type and
# help string.
#
# This must be included BEFORE any of the HIPFILE_<name> options are declared.
#
# To remove this compatibility layer, delete:
#   - This file
#   - The include() of this file in the top-level CMakeLists.txt
#   - The test/cmake/ais-compat directory
#   - The hipfile_ais_compat test in test/CMakeLists.txt
#-----------------------------------------------------------------------------

# The deprecated options. Each new name is the old name with the AIS_ prefix
# replaced by HIPFILE_.
set(HIPFILE_DEPRECATED_AIS_OPTIONS
    AIS_BUILD_DOCS
    AIS_CAPABLE_DIR
    AIS_CLANG_TIDY_EXECUTABLE
    AIS_CXX_STANDARD
    AIS_INSTALL_EXAMPLES
    AIS_INSTALL_TESTS
    AIS_INSTALL_TOOLS
    AIS_USE_CLANG_TIDY
    AIS_USE_CODE_COVERAGE
    AIS_USE_IWYU
    AIS_USE_SANITIZERS
    AIS_USE_THREAD_SANITIZER
    AIS_WARN_UNSAFE_BUFFER_OPS
)

# A function is used so the loop variables don't leak into the caller's scope.
# The cache is global, so the cache operations are still visible to the caller.
function(hipfile_map_deprecated_ais_options)
    foreach(old_name IN LISTS HIPFILE_DEPRECATED_AIS_OPTIONS)
        if(NOT DEFINED ${old_name})
            continue()
        endif()
        string(REGEX REPLACE "^AIS_" "HIPFILE_" new_name "${old_name}")
        message(DEPRECATION
            "${old_name} is deprecated, use ${new_name} instead. "
            "Setting ${new_name} to '${${old_name}}'.")
        set(${new_name} "${${old_name}}" CACHE STRING "" FORCE)
        set_property(CACHE ${new_name} PROPERTY TYPE UNINITIALIZED)
        unset(${old_name} CACHE)
    endforeach()
endfunction()

hipfile_map_deprecated_ais_options()
