# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT DEFINED MANIFEST_EXECUTABLE OR NOT DEFINED CAPABILITIES_DOCUMENT)
    message(
        FATAL_ERROR
        "MANIFEST_EXECUTABLE and CAPABILITIES_DOCUMENT are required"
    )
endif()

execute_process(
    COMMAND "${MANIFEST_EXECUTABLE}"
    RESULT_VARIABLE _manifest_result
    OUTPUT_VARIABLE _manifest
    ERROR_VARIABLE _manifest_error
)
if(NOT _manifest_result EQUAL 0)
    message(
        FATAL_ERROR
        "ConSan capability manifest failed (${_manifest_result}): "
        "${_manifest_error}"
    )
endif()

file(READ "${CAPABILITIES_DOCUMENT}" _document)
set(_begin "<!-- BEGIN GENERATED CONSAN CAPABILITY CONTRACT -->")
set(_end "<!-- END GENERATED CONSAN CAPABILITY CONTRACT -->")
string(FIND "${_document}" "${_begin}" _begin_offset)
string(FIND "${_document}" "${_end}" _end_offset)
if(_begin_offset LESS 0 OR _end_offset LESS _begin_offset)
    message(
        FATAL_ERROR
        "CAPABILITIES.md is missing the generated ConSan capability contract"
    )
endif()
string(LENGTH "${_end}" _end_length)
math(EXPR _block_length "${_end_offset} + ${_end_length} - ${_begin_offset}")
string(
    SUBSTRING "${_document}"
    ${_begin_offset}
    ${_block_length}
    _document_manifest
)
string(STRIP "${_manifest}" _manifest)
string(STRIP "${_document_manifest}" _document_manifest)
if(UPDATE)
    string(SUBSTRING "${_document}" 0 ${_begin_offset} _prefix)
    string(LENGTH "${_document}" _document_length)
    math(EXPR _suffix_offset "${_end_offset} + ${_end_length}")
    math(EXPR _suffix_length "${_document_length} - ${_suffix_offset}")
    string(SUBSTRING "${_document}" ${_suffix_offset} ${_suffix_length} _suffix)
    file(WRITE "${CAPABILITIES_DOCUMENT}" "${_prefix}${_manifest}${_suffix}")
    return()
endif()
if(NOT _manifest STREQUAL _document_manifest)
    set(_expected_path
        "${CMAKE_CURRENT_BINARY_DIR}/consan_capability_manifest.expected.md"
    )
    file(WRITE "${_expected_path}" "${_manifest}\n")
    message(
        FATAL_ERROR
        "CAPABILITIES.md does not match the typed ConSan capability contract.\n"
        "Run the consan_capability_manifest_update target, or compare "
        "${_expected_path}.\n"
        "--- expected ---\n${_manifest}\n"
        "--- found ---\n${_document_manifest}"
    )
endif()
