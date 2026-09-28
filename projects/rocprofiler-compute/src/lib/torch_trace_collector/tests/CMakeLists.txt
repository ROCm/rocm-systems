## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

# Only the real-header tests require PyTorch; the shipped collector does not.
set(ROCPROFCOMPUTE_TORCH_ROOT
    "${ROCPROFCOMPUTE_TORCH_ROOT}"
    CACHE PATH
    "PyTorch install used by test-torch-trace-collector"
)
set(_torch_root "${ROCPROFCOMPUTE_TORCH_ROOT}")
if(NOT _torch_root AND DEFINED ENV{ROCM_PATH})
    get_filename_component(_torch_root "$ENV{ROCM_PATH}/../torch" ABSOLUTE)
    if(NOT IS_DIRECTORY "${_torch_root}")
        set(_torch_root "")
    endif()
endif()
if(NOT _torch_root)
    message(
        STATUS
        "test-torch-trace-collector disabled: set ROCPROFCOMPUTE_TORCH_ROOT to enable"
    )
    return()
endif()
if(NOT EXISTS "${_torch_root}/include/ATen/ATen.h")
    message(
        FATAL_ERROR
        "PyTorch test install '${_torch_root}' has no include/ATen/ATen.h"
    )
endif()

set(_torch_libraries "")
foreach(_lib IN ITEMS torch torch_cpu c10)
    unset(_torch_test_library CACHE)
    find_library(
        _torch_test_library
        NAMES ${_lib}
        PATHS "${_torch_root}/lib"
        NO_DEFAULT_PATH
    )
    if(NOT _torch_test_library)
        message(FATAL_ERROR "lib${_lib} not found in ${_torch_root}/lib")
    endif()
    list(APPEND _torch_libraries "${_torch_test_library}")
endforeach()

set(target test-torch-trace-collector)
add_executable(${target} test_torch_trace_collector.cpp roctx_range_intercept.cpp)
target_link_libraries(
    ${target}
    PRIVATE
        torch-trace-collector
        gtest_main
        ${_torch_libraries}
        rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library
)
# Production includes are private; these translation units see real PyTorch.
target_include_directories(
    ${target}
    SYSTEM
    PRIVATE "${_torch_root}/include" "${_torch_root}/include/torch/csrc/api/include"
)
target_compile_features(${target} PRIVATE cxx_std_20)
target_compile_definitions(${target} PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_link_options(
    ${target}
    PRIVATE LINKER:--wrap=roctxRangePushA LINKER:--wrap=roctxRangePop
)

# ROCm wheels need the SDK's runtime libraries in addition to torch/lib.
set(_test_ld_paths "${_torch_root}/lib")
foreach(_rocm_root IN LISTS CMAKE_PREFIX_PATH)
    # Recent ROCm wheels also depend on the SDK's bundled system libraries.
    set(_sysdeps_dir "${_rocm_root}/lib/rocm_sysdeps/lib")
    if(IS_DIRECTORY "${_sysdeps_dir}")
        target_link_options(${target} PRIVATE "LINKER:-rpath-link,${_sysdeps_dir}")
        list(APPEND _test_ld_paths "${_sysdeps_dir}")
    endif()
    foreach(_suffix IN ITEMS lib/host-math/lib lib lib64)
        if(IS_DIRECTORY "${_rocm_root}/${_suffix}")
            target_link_directories(${target} PRIVATE "${_rocm_root}/${_suffix}")
            list(APPEND _test_ld_paths "${_rocm_root}/${_suffix}")
        endif()
    endforeach()
endforeach()
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
