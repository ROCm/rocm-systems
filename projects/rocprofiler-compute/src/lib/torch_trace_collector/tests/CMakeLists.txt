## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

set(runtime_test_target test-torch-abi-runtime)
add_executable(${runtime_test_target} test_torch_abi_runtime.cpp ../torch_abi/runtime.cpp)
target_include_directories(
    ${runtime_test_target}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../torch_abi
)
target_link_libraries(${runtime_test_target} PRIVATE ${CMAKE_DL_LIBS} gtest_main)
target_compile_features(${runtime_test_target} PRIVATE cxx_std_17)
target_compile_definitions(${runtime_test_target} PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

if(INSTALL_TESTS)
    install(
        TARGETS ${runtime_test_target}
        RUNTIME
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${CMAKE_PROJECT_NAME}/tests
            COMPONENT tests
    )
endif()

# The production collector always builds without PyTorch. Real-libTorch tests
# are enabled only when a matching development installation is available.
set(_torch_include_dirs "")
set(_torch_libraries "")
set(_torch_lib_dir "")

if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    get_filename_component(_rocm_path "$ENV{ROCM_PATH}" ABSOLUTE)
    get_filename_component(_torch_prefix "${_rocm_path}/../torch" ABSOLUTE)

    if(EXISTS "${_torch_prefix}/include/ATen/ATen.h")
        set(_torch_lib_dir "${_torch_prefix}/lib")
        set(_torch_include_dirs
            "${_torch_prefix}/include"
            "${_torch_prefix}/include/torch/csrc/api/include"
        )
        foreach(_lib IN ITEMS torch torch_cpu c10)
            find_library(
                TORCH_${_lib}_LIBRARY
                NAMES ${_lib}
                PATHS "${_torch_lib_dir}"
                NO_DEFAULT_PATH
            )
            if(NOT TORCH_${_lib}_LIBRARY)
                message(WARNING "lib${_lib} not found in ${_torch_lib_dir}")
                set(_torch_include_dirs "")
                set(_torch_libraries "")
                break()
            endif()
            list(APPEND _torch_libraries "${TORCH_${_lib}_LIBRARY}")
        endforeach()
    endif()
endif()

if(NOT _torch_include_dirs)
    message(
        STATUS
        "No PyTorch development install found; test-torch-trace-collector is disabled"
    )
    return()
endif()

set(target test-torch-trace-collector)

add_executable(${target} test_torch_trace_collector.cpp roctx_range_intercept.cpp)

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
