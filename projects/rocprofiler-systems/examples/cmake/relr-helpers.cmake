# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

#
# Helpers for building example executables with RELR (compact relative)
# relocations. RELR packs relative relocations into a compact .relr.dyn
# section instead of individual R_*_RELATIVE entries in .rela.dyn, which is
# useful for exercising the Dyninst RELR packing support.
#
# Enabling RELR requires:
#   * position-independent code / a PIE (or shared) output
#   * a linker that understands "-z pack-relative-relocs"
#
include_guard(DIRECTORY)

include(CheckLinkerFlag)

option(
    ROCPROFSYS_EXAMPLES_USE_RELR
    "Link example executables with RELR (compact relative) relocations"
    ON
)

set(ROCPROFSYS_RELR_LINK_FLAG "-Wl,-z,pack-relative-relocs")

if(ROCPROFSYS_EXAMPLES_USE_RELR AND NOT DEFINED ROCPROFSYS_LINKER_HAS_RELR)
    check_linker_flag(CXX "${ROCPROFSYS_RELR_LINK_FLAG}" ROCPROFSYS_LINKER_HAS_RELR)
    if(NOT ROCPROFSYS_LINKER_HAS_RELR)
        message(
            AUTHOR_WARNING
            "ROCPROFSYS_EXAMPLES_USE_RELR is ON but the linker does not support "
            "'${ROCPROFSYS_RELR_LINK_FLAG}'. Examples will be built without RELR "
            "relocations."
        )
    endif()
endif()

# Enable RELR relocations on one or more example targets. Silently does nothing
# when RELR is disabled or unsupported, so it is safe to call unconditionally.
function(rocprofiler_systems_example_enable_relr)
    if(NOT ROCPROFSYS_EXAMPLES_USE_RELR OR NOT ROCPROFSYS_LINKER_HAS_RELR)
        return()
    endif()

    foreach(_target IN LISTS ARGN)
        if(NOT TARGET ${_target})
            continue()
        endif()

        set_property(TARGET ${_target} PROPERTY POSITION_INDEPENDENT_CODE ON)
        target_link_options(${_target} PRIVATE -pie ${ROCPROFSYS_RELR_LINK_FLAG})
    endforeach()
endfunction()
