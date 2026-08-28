## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

set(target test-torch-trace-collector)

add_executable(${target} test_torch_trace_collector.cpp roctx_range_intercept.cpp)

target_link_libraries(${target} PRIVATE torch-trace-collector gtest_main)
target_link_options(
    ${target}
    PRIVATE LINKER:--allow-shlib-undefined LINKER:--wrap=roctxRangePushA
)

get_filename_component(_roctx_lib_dir "${ROCTX_LIBRARY}" DIRECTORY)
set_target_properties(
    ${target}
    PROPERTIES INSTALL_RPATH "${_torch_lib_dir};${_roctx_lib_dir}"
)

set(_openblas_dir "")
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
    file(GLOB _openblas_hits "${_dir}/librocm-openblas.so*")
    if(_openblas_hits)
        set(_openblas_dir "${_dir}")
        break()
    endif()
endforeach()

set(_test_ld_paths "${_torch_lib_dir}")
if(_openblas_dir)
    list(APPEND _test_ld_paths "${_openblas_dir}")
endif()
list(JOIN _test_ld_paths ":" _test_ld_path)

set_target_properties(
    ${target}
    PROPERTIES ROCPROF_TEST_LD_LIBRARY_PATH "${_test_ld_path}"
)

if(INSTALL_TESTS)
    install(
        TARGETS ${target}
        RUNTIME
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${CMAKE_PROJECT_NAME}/tests
            COMPONENT tests
    )
endif()
