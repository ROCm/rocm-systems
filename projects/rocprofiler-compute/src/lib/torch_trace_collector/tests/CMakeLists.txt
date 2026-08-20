## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

set(target test-torch-trace-collector)

add_executable(
    ${target}
    test_torch_trace_collector.cpp
    roctx_range_intercept.cpp
)

# Includes, torch libraries, and the DebugInfoKind flag are inherited from
# torch-trace-collector. Only the runtime search paths below are test-specific.
target_link_libraries(${target} PRIVATE torch-trace-collector gtest_main)

# torch's shared libraries cross-reference one another at load time.
target_link_options(
    ${target}
    PRIVATE LINKER:--allow-shlib-undefined LINKER:--wrap=roctxRangePushA
)

get_filename_component(_roctx_lib_dir "${ROCTX_LIB}" DIRECTORY)
set_target_properties(
    ${target}
    PROPERTIES INSTALL_RPATH "${_torch_lib_search_dirs};${_roctx_lib_dir}"
)

# On some ROCm layouts openblas is outside the torch wheel; add it to the
# runtime library path when present.
set(_ttc_openblas_dir "")
foreach(
    _dir
    IN
    ITEMS
        "$ENV{ROCM_PATH}/lib/host-math/lib"
        "/opt/rocm/lib/host-math/lib"
        "$ENV{ROCM_PATH}/lib"
        "/opt/rocm/lib"
        "$ENV{ROCM_PATH}/lib64"
        "/opt/rocm/lib64"
)
    file(GLOB _ttc_openblas_hits "${_dir}/librocm-openblas.so*")
    if(_ttc_openblas_hits)
        set(_ttc_openblas_dir "${_dir}")
        break()
    endif()
endforeach()

set(_ttc_test_ld_paths "${_torch_lib_search_dirs}")
if(_ttc_openblas_dir)
    list(APPEND _ttc_test_ld_paths "${_ttc_openblas_dir}")
endif()
list(JOIN _ttc_test_ld_paths ":" _ttc_test_ld_path)

if(TEST_FROM_INSTALL)
    set(_ttc_test_cmd tests/${target})
else()
    set(_ttc_test_cmd $<TARGET_FILE:${target}>)
endif()
add_test(NAME ${target} COMMAND ${_ttc_test_cmd})
set_tests_properties(
    ${target}
    PROPERTIES
        LABELS "profile"
        ENVIRONMENT "LD_LIBRARY_PATH=${_ttc_test_ld_path}:$ENV{LD_LIBRARY_PATH}"
)

if(INSTALL_TESTS)
    install(
        TARGETS ${target}
        RUNTIME
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/rocprofiler-compute/tests
            COMPONENT tests
    )
endif()
