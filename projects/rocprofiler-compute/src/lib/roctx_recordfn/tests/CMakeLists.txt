## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

set(target test-roctx-recordfn)

find_library(
    _rcfn_libpython
    NAMES python${_py_major}.${_py_minor} python${_py_major}.${_py_minor}m
    HINTS ${Python3_LIBRARY_DIRS}
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
)
find_path(
    _rcfn_python_include
    NAMES Python.h
    HINTS ${Python3_INCLUDE_DIRS}
    PATHS
        /usr/include/python${_py_major}.${_py_minor}
        /usr/local/include/python${_py_major}.${_py_minor}
)
if(NOT _rcfn_libpython OR NOT _rcfn_python_include)
    message(
        WARNING
        "roctx_recordfn: skipping ${target}: Python ${_py_major}.${_py_minor} "
        "development files (headers + libpython) not found. Install the "
        "python development package for your distro to enable this test "
        "(e.g. libpython${_py_major}.${_py_minor}-dev on Debian/Ubuntu, "
        "python${_py_major}${_py_minor}-devel on RHEL/Fedora)."
    )
    return()
endif()

add_executable(${target} test_roctx_recordfn.cpp)

target_link_libraries(
    ${target}
    PRIVATE gtest_main ${_torch_resolved_libs} ${ROCTX_LIB} ${_rcfn_libpython}
)

target_link_options(${target} PRIVATE LINKER:--allow-shlib-undefined)

target_include_directories(
    ${target}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${_torch_includes}
        ${ROCTX_INCLUDE_DIR}
        ${_rcfn_python_include}
)

if(ROCPROF_TORCHTRACE_HAS_CUSTOM_DBGINFOKIND)
    target_compile_definitions(
        ${target}
        PRIVATE ROCPROF_TORCHTRACE_HAS_CUSTOM_DBGINFOKIND=1
    )
endif()

get_filename_component(_roctx_lib_dir "${ROCTX_LIB}" DIRECTORY)

set_target_properties(
    ${target}
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        INSTALL_RPATH "${_torch_lib_search_dirs};${_roctx_lib_dir}"
)

set(_rcfn_openblas_dir "")
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
    file(GLOB _rcfn_openblas_hits "${_dir}/librocm-openblas.so*")
    if(_rcfn_openblas_hits)
        list(SORT _rcfn_openblas_hits COMPARE NATURAL ORDER DESCENDING)
        list(GET _rcfn_openblas_hits 0 _rcfn_openblas_latest)
        set(_rcfn_openblas_dir "${_dir}")
        message(STATUS "roctx_recordfn: using openblas ${_rcfn_openblas_latest}")
        break()
    endif()
endforeach()

set(_rcfn_test_ld_paths "${_torch_lib_search_dirs}")
if(_rcfn_openblas_dir)
    list(APPEND _rcfn_test_ld_paths "${_rcfn_openblas_dir}")
endif()
list(JOIN _rcfn_test_ld_paths ":" _rcfn_test_ld_path)

if(TEST_FROM_INSTALL)
    set(_rcfn_test_cmd tests/${target})
else()
    set(_rcfn_test_cmd $<TARGET_FILE:${target}>)
endif()
add_test(NAME ${target} COMMAND ${_rcfn_test_cmd})
set_tests_properties(
    ${target}
    PROPERTIES
        LABELS "profile"
        ENVIRONMENT "LD_LIBRARY_PATH=${_rcfn_test_ld_path}:$ENV{LD_LIBRARY_PATH}"
)

if(INSTALL_TESTS)
    install(
        TARGETS ${target}
        RUNTIME
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${CMAKE_PROJECT_NAME}/tests
            COMPONENT tests
    )
endif()
