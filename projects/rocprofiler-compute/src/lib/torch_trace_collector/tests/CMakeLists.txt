## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

set(target test-torch-trace-collector)

add_executable(${target} test_torch_trace_collector.cpp roctx_range_intercept.cpp)

target_link_libraries(${target} PRIVATE torch-trace-collector gtest_main)
# roctx_range_intercept.cpp records range strings via __wrap_roctxRangePushA.
target_link_options(${target} PRIVATE LINKER:--wrap=roctxRangePushA)

# libtorch_cpu needs rocm-openblas but its RUNPATH points at the wheel builder.
set(_openblas_dir "")
foreach(_rocm_root IN LISTS CMAKE_PREFIX_PATH)
    foreach(_suffix IN ITEMS lib/host-math/lib lib lib64)
        file(GLOB _openblas_hits "${_rocm_root}/${_suffix}/librocm-openblas.so*")
        if(_openblas_hits)
            set(_openblas_dir "${_rocm_root}/${_suffix}")
            break()
        endif()
    endforeach()
    if(_openblas_dir)
        break()
    endif()
endforeach()
if(_openblas_dir)
    target_link_directories(${target} PRIVATE "${_openblas_dir}")
endif()

# libtorch is outside the ROCm tree, so ctest passes it on LD_LIBRARY_PATH.
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
