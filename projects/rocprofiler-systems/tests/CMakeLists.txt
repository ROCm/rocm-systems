# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

#
# rocprofiler-systems tests
#
include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/pytest/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-pytest.cmake)

# ------------------------------------------------------------------------------#
#
# Perfetto trace_processor_shell
#
# To be used only by the build tree, not installed
#
# ------------------------------------------------------------------------------#

rocprofiler_systems_add_option(
    ROCPROFSYS_DOWNLOAD_TRACE_PROCESSOR_SHELL
    "Download a pinned perfetto trace_processor_shell for the test suite"
    ON
    NO_FEATURE
)
rocprofiler_systems_add_cache_option(
    ROCPROFSYS_TRACE_PROCESSOR_SHELL_URL
    "URL of the prebuilt perfetto trace_processor_shell used by the test suite"
    STRING
    "https://commondatastorage.googleapis.com/perfetto-luci-artifacts/v47.0/linux-amd64/trace_processor_shell"
    NO_FEATURE
)
rocprofiler_systems_add_cache_option(
    ROCPROFSYS_TRACE_PROCESSOR_SHELL_SHA256
    "Expected SHA256 of ROCPROFSYS_TRACE_PROCESSOR_SHELL_URL"
    STRING
    "832425c3c7934904d1e0ec1721beb51423de7dbcf399a899973f2b6b464603fa"
    NO_FEATURE
)
rocprofiler_systems_add_cache_option(
    ROCPROFSYS_TRACE_PROCESSOR_SHELL
    "Existing trace_processor_shell to stage instead of downloading one"
    FILEPATH
    ""
    NO_FEATURE
)

set(_trace_processor_shell
    "${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/trace_processor_shell"
)
set(_trace_processor_shell_permissions
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE
)

# the pinned binary is x86-64; other architectures need their own URL and checksum
set(_download_trace_processor_shell ${ROCPROFSYS_DOWNLOAD_TRACE_PROCESSOR_SHELL})
if(
    "${ROCPROFSYS_TRACE_PROCESSOR_SHELL_URL}" MATCHES "linux-amd64"
    AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$"
)
    set(_download_trace_processor_shell OFF)
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests")

if(ROCPROFSYS_TRACE_PROCESSOR_SHELL)
    if(EXISTS "${ROCPROFSYS_TRACE_PROCESSOR_SHELL}")
        file(
            COPY_FILE "${ROCPROFSYS_TRACE_PROCESSOR_SHELL}"
            "${_trace_processor_shell}"
            ONLY_IF_DIFFERENT
        )
        file(
            CHMOD
            "${_trace_processor_shell}"
            PERMISSIONS ${_trace_processor_shell_permissions}
        )
    else()
        rocprofiler_systems_message(
            WARNING
            "ROCPROFSYS_TRACE_PROCESSOR_SHELL does not exist: ${ROCPROFSYS_TRACE_PROCESSOR_SHELL}. Perfetto validation will download a trace_processor_shell on demand."
        )
    endif()
elseif(_download_trace_processor_shell)
    # a previously staged binary is reused only when its contents still match
    if(EXISTS "${_trace_processor_shell}")
        file(SHA256 "${_trace_processor_shell}" _sha256)
        if(NOT "${_sha256}" STREQUAL "${ROCPROFSYS_TRACE_PROCESSOR_SHELL_SHA256}")
            file(REMOVE "${_trace_processor_shell}")
        endif()
    endif()

    if(NOT EXISTS "${_trace_processor_shell}")
        rocprofiler_systems_message(
            STATUS
            "Downloading ${ROCPROFSYS_TRACE_PROCESSOR_SHELL_URL}"
        )
        file(
            DOWNLOAD "${ROCPROFSYS_TRACE_PROCESSOR_SHELL_URL}"
            "${_trace_processor_shell}.tmp"
            STATUS _download_status
            TLS_VERIFY ON
            INACTIVITY_TIMEOUT 10
            TIMEOUT 40
        )
        list(GET _download_status 0 _download_code)
        list(GET _download_status 1 _download_message)

        set(_download_error "")
        if(NOT _download_code EQUAL 0)
            set(_download_error "${_download_message}")
        else()
            file(SHA256 "${_trace_processor_shell}.tmp" _sha256)
            if("${_sha256}" STREQUAL "${ROCPROFSYS_TRACE_PROCESSOR_SHELL_SHA256}")
                file(RENAME "${_trace_processor_shell}.tmp" "${_trace_processor_shell}")
                file(
                    CHMOD
                    "${_trace_processor_shell}"
                    PERMISSIONS ${_trace_processor_shell_permissions}
                )
            else()
                set(_download_error
                    "expected SHA256 ${ROCPROFSYS_TRACE_PROCESSOR_SHELL_SHA256}, got ${_sha256}"
                )
            endif()
        endif()

        if(_download_error)
            file(REMOVE "${_trace_processor_shell}.tmp")
            rocprofiler_systems_message(
                WARNING
                "Could not stage a trace_processor_shell from ${ROCPROFSYS_TRACE_PROCESSOR_SHELL_URL} (${_download_error}). Perfetto validation will download one on demand, which is slower and can fail on hosts without network access."
            )
        endif()
    endif()
endif()

# ------------------------------------------------------------------------------#
#
# Move test files to build directory
#
# ------------------------------------------------------------------------------#

set(ROCPROFSYS_PYTHON_VALIDATION_FILES
    ${CMAKE_CURRENT_LIST_DIR}/check_amd_smi_metrics.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-causal-json.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-perfetto-proto.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-rocpd.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-timemory-json.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-unified-memory.py
)

set(ROCPROFSYS_TEST_SCRIPTS
    ${CMAKE_CURRENT_LIST_DIR}/get_default_nic.sh
    ${CMAKE_CURRENT_LIST_DIR}/generate_papi_nic_events.sh
    ${CMAKE_CURRENT_LIST_DIR}/run_rocprofiler_systems.py
)

set(_capchk_dir ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests)
add_executable(
    rocprof-sys-capchk
    EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-capchk.cpp
)
set_target_properties(
    rocprof-sys-capchk
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY "${_capchk_dir}"
)

# test_categories.yaml drives tier (quick/standard/comprehensive/full) label
# injection at CTest-generate time (see conftest.py:_load_test_categories).
set(ROCPROFSYS_TEST_CATEGORIES_YAML ${CMAKE_CURRENT_LIST_DIR}/test_categories.yaml)

add_custom_target(
    copy-test-files
    ALL
    COMMAND
        ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${ROCPROFSYS_PYTHON_VALIDATION_FILES}
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_LIST_DIR}/README.md
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${ROCPROFSYS_TEST_CATEGORIES_YAML}
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/rocpd-validation-rules
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${ROCPROFSYS_TEST_SCRIPTS}
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests
)
add_dependencies(copy-test-files rocprof-sys-capchk)
add_dependencies(generate-pytest-ctests copy-test-files)

# ------------------------------------------------------------------------------#
#
# Pytests install
#
# ------------------------------------------------------------------------------#

if(ROCPROFSYS_INSTALL_TESTING)
    # Python Validation scripts
    install(
        PROGRAMS ${ROCPROFSYS_PYTHON_VALIDATION_FILES}
        DESTINATION share/rocprofiler-systems/tests
        COMPONENT rocprofiler-systems-tests
    )
    install(
        DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules
        DESTINATION share/rocprofiler-systems/tests
        COMPONENT rocprofiler-systems-tests
    )
    # Scripts
    install(
        PROGRAMS ${ROCPROFSYS_TEST_SCRIPTS}
        DESTINATION share/rocprofiler-systems/tests
        COMPONENT rocprofiler-systems-tests
    )
    # Capability check binary
    install(
        TARGETS rocprof-sys-capchk
        RUNTIME
            DESTINATION share/rocprofiler-systems/tests
            COMPONENT rocprofiler-systems-tests
    )
    # Test categories policy
    install(
        FILES ${ROCPROFSYS_TEST_CATEGORIES_YAML}
        DESTINATION share/rocprofiler-systems/tests
        COMPONENT rocprofiler-systems-tests
    )
endif()
