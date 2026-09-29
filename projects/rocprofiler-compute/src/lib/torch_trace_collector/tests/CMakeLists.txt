## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

# The collector itself builds without PyTorch. This test links the real
# libtorch so it can check the torch_abi.h layouts against the real headers,
# so it is built only where PyTorch is installed. rocprofiler-compute requires
# PyTorch in the same prefix as ROCm, which is the default probed below.
if(NOT ROCPROFCOMPUTE_TORCH_ROOT AND DEFINED ENV{ROCM_PATH})
    get_filename_component(_rocm_path "$ENV{ROCM_PATH}" ABSOLUTE)
    get_filename_component(ROCPROFCOMPUTE_TORCH_ROOT "${_rocm_path}/../torch" ABSOLUTE)
endif()
set(ROCPROFCOMPUTE_TORCH_ROOT
    "${ROCPROFCOMPUTE_TORCH_ROOT}"
    CACHE PATH
    "PyTorch install used by test-torch-trace-collector"
)

if(NOT IS_DIRECTORY "${ROCPROFCOMPUTE_TORCH_ROOT}")
    message(
        STATUS
        "test-torch-trace-collector is disabled: no PyTorch install at "
        "'${ROCPROFCOMPUTE_TORCH_ROOT}'. Set ROCPROFCOMPUTE_TORCH_ROOT to enable it."
    )
    return()
endif()

# The directory exists, so an incomplete install is a broken layout rather
# than an absent one. Fail instead of dropping the test.
set(_torch_lib_dir "${ROCPROFCOMPUTE_TORCH_ROOT}/lib")
set(_torch_include_dirs
    "${ROCPROFCOMPUTE_TORCH_ROOT}/include"
    "${ROCPROFCOMPUTE_TORCH_ROOT}/include/torch/csrc/api/include"
)
if(NOT EXISTS "${ROCPROFCOMPUTE_TORCH_ROOT}/include/ATen/ATen.h")
    message(
        FATAL_ERROR
        "ROCPROFCOMPUTE_TORCH_ROOT='${ROCPROFCOMPUTE_TORCH_ROOT}' has no "
        "include/ATen/ATen.h"
    )
endif()
set(_torch_libraries "")
foreach(_lib IN ITEMS torch torch_cpu c10)
    find_library(
        TORCH_${_lib}_LIBRARY
        NAMES ${_lib}
        PATHS "${_torch_lib_dir}"
        NO_DEFAULT_PATH
    )
    if(NOT TORCH_${_lib}_LIBRARY)
        message(FATAL_ERROR "lib${_lib} not found in ${_torch_lib_dir}")
    endif()
    list(APPEND _torch_libraries "${TORCH_${_lib}_LIBRARY}")
endforeach()

set(target test-torch-trace-collector)

# torch_trace_collector_module.cpp brings in the plain-C entry points so the
# shipped interface is exercised against the real libtorch.
add_executable(
    ${target}
    test_torch_trace_collector.cpp
    roctx_range_intercept.cpp
    ../torch_trace_collector_module.cpp
)

target_link_libraries(${target} PRIVATE torch-trace-collector gtest_main)
# roctx_range_intercept.cpp records range strings via __wrap_roctxRangePushA.
target_link_options(${target} PRIVATE LINKER:--wrap=roctxRangePushA)

target_include_directories(${target} SYSTEM PRIVATE ${_torch_include_dirs})
target_link_libraries(${target} PRIVATE ${_torch_libraries})
target_compile_features(${target} PRIVATE cxx_std_20)

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

set(_test_ld_paths "")
list(APPEND _test_ld_paths "${_torch_lib_dir}")
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
