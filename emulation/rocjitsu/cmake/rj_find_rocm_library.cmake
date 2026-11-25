# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Resolve a runtime library from the selected ROCM_PATH on every configure.
# A cached find_library result from another installation can otherwise mix
# compiler, headers, and runtime libraries within one HIP test build.
function(rj_find_rocm_library out_var library_name)
    find_library(
        _rj_selected_library
        NAMES ${library_name}
        PATHS "${ROCM_PATH}"
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH
        NO_CACHE
    )
    set(${out_var}
        "${_rj_selected_library}"
        CACHE FILEPATH
        "${library_name} from the selected ROCM_PATH"
        FORCE
    )
    set(${out_var} "${_rj_selected_library}" PARENT_SCOPE)
endfunction()
