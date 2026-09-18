## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

# Always built. The tests that drive libtorch need the real headers, so they
# compile only when a PyTorch install is found beside ROCM_PATH.
set(_torch_include_dirs "")
set(_torch_libraries "")
set(_torch_lib_dir "")

if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    get_filename_component(_rocm_path "$ENV{ROCM_PATH}" ABSOLUTE)
    get_filename_component(_torch_prefix "${_rocm_path}/../torch" ABSOLUTE)

    set(_torch_version_headers
        "${_torch_prefix}/include/torch/headeronly/version.h"
        "${_torch_prefix}/include/torch/csrc/api/include/torch/version.h"
    )
    set(_torch_version "")
    foreach(_header IN LISTS _torch_version_headers)
        if(NOT EXISTS "${_header}")
            continue()
        endif()
        file(STRINGS "${_header}" _major_line REGEX "define[ \t]+TORCH_VERSION_MAJOR")
        file(STRINGS "${_header}" _minor_line REGEX "define[ \t]+TORCH_VERSION_MINOR")
        if(_major_line AND _minor_line)
            string(REGEX REPLACE "[^0-9]" "" _torch_major "${_major_line}")
            string(REGEX REPLACE "[^0-9]" "" _torch_minor "${_minor_line}")
            set(_torch_version "${_torch_major}.${_torch_minor}")
            break()
        endif()
    endforeach()

    # Matches kSupportedTorchVersions in torch_abi.h. Checking the stub numbers
    # against a version they do not describe would only produce noise.
    if(_torch_version MATCHES "^2\\.(13|14)$")
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
    elseif(_torch_version)
        message(
            STATUS
            "PyTorch ${_torch_version} is not described by torch_abi.h; "
            "building test-torch-trace-collector without the libtorch tests"
        )
    endif()
endif()

if(NOT _torch_include_dirs)
    message(
        STATUS
        "No supported PyTorch install found; test-torch-trace-collector will "
        "build without its libtorch tests"
    )
endif()

set(target test-torch-trace-collector)

add_executable(${target} test_torch_trace_collector.cpp roctx_range_intercept.cpp)

target_link_libraries(${target} PRIVATE torch-trace-collector gtest_main)
# roctx_range_intercept.cpp records range strings via __wrap_roctxRangePushA.
target_link_options(${target} PRIVATE LINKER:--wrap=roctxRangePushA)

if(_torch_include_dirs)
    target_include_directories(${target} SYSTEM PRIVATE ${_torch_include_dirs})
    target_link_libraries(${target} PRIVATE ${_torch_libraries})
else()
    # The collector's own headers include ATen, so the stubs stand in here too.
    target_include_directories(
        ${target}
        SYSTEM
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../torch_abi
    )
    # The collector's ATen and c10 references are left dangling. Nothing that
    # calls them is compiled in without the headers.
    target_link_options(${target} PRIVATE LINKER:--unresolved-symbols=ignore-all)
endif()

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
if(_openblas_dir AND _torch_libraries)
    target_link_directories(${target} PRIVATE "${_openblas_dir}")
endif()

# libtorch is outside the ROCm tree, so ctest passes it on LD_LIBRARY_PATH.
set(_test_ld_paths "")
if(_torch_lib_dir)
    list(APPEND _test_ld_paths "${_torch_lib_dir}")
endif()
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
