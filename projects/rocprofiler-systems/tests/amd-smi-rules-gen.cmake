# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#
# amd-smi-rules-gen.cmake
#
# This module detects hardware configuration using rocminfo and amd-smi metric,
# then generates a combined JSON validation rules file based on detected capabilities.
# Output format matches amd-smi-rules.json for use with validate-rocpd.py
#

include_guard(GLOBAL)

# ---------------------------------------------------------------------------- #
# Get rocminfo output
# ---------------------------------------------------------------------------- #
function(DETECT_ROCMINFO OUTPUT_VAR)
    # Check if we already have cached rocminfo
    if(
        DEFINED CACHE{_ROCMINFO_CACHED_OUTPUT}
        AND NOT "${_ROCMINFO_CACHED_OUTPUT}" STREQUAL ""
    )
        rocprofiler_systems_message(STATUS "Using cached rocminfo output")
        set(${OUTPUT_VAR} "${_ROCMINFO_CACHED_OUTPUT}" PARENT_SCOPE)
        return()
    endif()

    # Check if rocminfo_EXECUTABLE is already defined
    if(NOT rocminfo_EXECUTABLE)
        find_program(
            rocminfo_EXECUTABLE
            NAMES rocminfo
            HINTS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm
            PATHS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm
            PATH_SUFFIXES bin
        )
    endif()

    if(NOT rocminfo_EXECUTABLE)
        rocprofiler_systems_message(WARNING "rocminfo not found")
        set(${OUTPUT_VAR} "" PARENT_SCOPE)
        return()
    endif()

    execute_process(
        COMMAND ${rocminfo_EXECUTABLE}
        OUTPUT_VARIABLE _ROCMINFO_OUTPUT
        ERROR_VARIABLE _ROCMINFO_ERROR
        RESULT_VARIABLE _ROCMINFO_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
        TIMEOUT 30
    )

    if(NOT _ROCMINFO_RESULT EQUAL 0)
        rocprofiler_systems_message(
            WARNING "rocminfo execution failed: ${_ROCMINFO_ERROR}"
        )
        set(${OUTPUT_VAR} "" PARENT_SCOPE)
        return()
    endif()

    # Cache the output to avoid running rocminfo multiple times
    set(_ROCMINFO_CACHED_OUTPUT
        "${_ROCMINFO_OUTPUT}"
        CACHE INTERNAL
        "Cached rocminfo output"
    )

    set(${OUTPUT_VAR} "${_ROCMINFO_OUTPUT}" PARENT_SCOPE)
    rocprofiler_systems_message(
        STATUS "rocminfo executed successfully (using ${rocminfo_EXECUTABLE})"
    )
endfunction()

# ---------------------------------------------------------------------------- #
# Run amd-smi metric and use info to find supported metrics
# ---------------------------------------------------------------------------- #
function(DETECT_AMD_SMI_METRIC OUTPUT_VAR)
    if(
        DEFINED CACHE{_AMD_SMI_CACHED_OUTPUT}
        AND NOT "${_AMD_SMI_CACHED_OUTPUT}" STREQUAL ""
    )
        rocprofiler_systems_message(STATUS "Using cached amd-smi metric output")
        set(${OUTPUT_VAR} "${_AMD_SMI_CACHED_OUTPUT}" PARENT_SCOPE)
        return()
    endif()

    if(ROCPROFSYS_AMD_SMI_EXE)
        set(_AMD_SMI_EXE "${ROCPROFSYS_AMD_SMI_EXE}")
    elseif(NOT AMD_SMI_EXECUTABLE)
        find_program(
            AMD_SMI_EXECUTABLE
            NAMES amd-smi
            HINTS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm
            PATHS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm
            PATH_SUFFIXES bin
        )
        set(_AMD_SMI_EXE "${AMD_SMI_EXECUTABLE}")
    else()
        set(_AMD_SMI_EXE "${AMD_SMI_EXECUTABLE}")
    endif()

    if(NOT _AMD_SMI_EXE)
        rocprofiler_systems_message(WARNING "amd-smi not found")
        set(${OUTPUT_VAR} "" PARENT_SCOPE)
        return()
    endif()

    execute_process(
        COMMAND ${_AMD_SMI_EXE} metric
        OUTPUT_VARIABLE _AMD_SMI_OUTPUT
        ERROR_VARIABLE _AMD_SMI_ERROR
        RESULT_VARIABLE _AMD_SMI_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        TIMEOUT 30
    )

    if(NOT _AMD_SMI_RESULT EQUAL 0)
        rocprofiler_systems_message(
            WARNING "amd-smi metric execution failed: ${_AMD_SMI_ERROR}"
        )
        set(${OUTPUT_VAR} "" PARENT_SCOPE)
        return()
    endif()

    # Cache the output to avoid running amd-smi multiple times
    set(_AMD_SMI_CACHED_OUTPUT
        "${_AMD_SMI_OUTPUT}"
        CACHE INTERNAL
        "Cached amd-smi metric output"
    )

    set(${OUTPUT_VAR} "${_AMD_SMI_OUTPUT}" PARENT_SCOPE)
    rocprofiler_systems_message(STATUS "amd-smi metric executed successfully")
endfunction()

# ---------------------------------------------------------------------------- #
# Parse rocminfo output to count CPUs and GPUs/APUs
# ---------------------------------------------------------------------------- #
function(PARSE_ROCMINFO_CPU_COUNT ROCMINFO_OUTPUT CPU_COUNT_VAR GPU_COUNT_VAR)
    set(_CPU_COUNT 0)
    set(_GPU_COUNT 0)

    # Split output into lines
    string(REPLACE "\n" ";" _LINES "${ROCMINFO_OUTPUT}")

    foreach(_LINE IN LISTS _LINES)
        # Check for CPU Device Type
        if(_LINE MATCHES "Device Type:.*CPU")
            math(EXPR _CPU_COUNT "${_CPU_COUNT} + 1")
        endif()
        # Check for GPU Device Type
        if(_LINE MATCHES "Device Type:.*GPU")
            math(EXPR _GPU_COUNT "${_GPU_COUNT} + 1")
        endif()
    endforeach()

    set(${CPU_COUNT_VAR} ${_CPU_COUNT} PARENT_SCOPE)
    set(${GPU_COUNT_VAR} ${_GPU_COUNT} PARENT_SCOPE)

    rocprofiler_systems_message(
        STATUS "Found ${_CPU_COUNT} CPU(s), ${_GPU_COUNT} GPU(s)"
    )
endfunction()

# ---------------------------------------------------------------------------- #
# Helper function: Check if a metric value is valid (not N/A).
# Returns TRUE if valid, FALSE if N/A or empty.
# Handles both single values and arrays like [0 %, N/A, N/A].
# ---------------------------------------------------------------------------- #
function(_IS_METRIC_VALUE_VALID VALUE RESULT_VAR)
    set(_IS_VALID FALSE)

    # Check for empty
    if("${VALUE}" STREQUAL "")
        set(${RESULT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()

    # Check for direct N/A
    if("${VALUE}" STREQUAL "N/A")
        set(${RESULT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()

    # Check for array starting with [N/A (all N/A values)
    if("${VALUE}" MATCHES "^\\[N/A")
        set(${RESULT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()

    # Check if array has at least one non-N/A value
    # e.g., [0 %, N/A, N/A] should be valid
    if("${VALUE}" MATCHES "^\\[")
        # It's an array - check if ALL values are N/A
        string(REGEX REPLACE "\\[|\\]" "" _ARRAY_CONTENT "${VALUE}")
        string(REPLACE "," ";" _ARRAY_ITEMS "${_ARRAY_CONTENT}")
        set(_HAS_VALID_ITEM FALSE)
        foreach(_ITEM IN LISTS _ARRAY_ITEMS)
            string(STRIP "${_ITEM}" _ITEM)
            if(NOT "${_ITEM}" STREQUAL "N/A" AND NOT "${_ITEM}" STREQUAL "")
                set(_HAS_VALID_ITEM TRUE)
                break()
            endif()
        endforeach()
        set(${RESULT_VAR} ${_HAS_VALID_ITEM} PARENT_SCOPE)
        return()
    endif()

    # Value is valid (numeric or string that's not N/A)
    set(${RESULT_VAR} TRUE PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------- #
# Parse amd-smi metric output to get supported metrics for each GPU.
# ---------------------------------------------------------------------------- #
function(PARSE_AMD_SMI_METRICS AMD_SMI_OUTPUT GPU_COUNT_VAR GPU_METRICS_VAR)
    set(_GPU_COUNT 0)
    set(_GPU_METRICS "")
    set(_CURRENT_GPU -1)
    set(_CURRENT_METRICS "")

    # Define supported metric categories based on rocprof-sys amd_smi.cpp
    set(_METRIC_CATEGORIES
        "GFX_ACTIVITY"
        "EDGE"
        "HOTSPOT"
        "SOCKET_POWER"
        "TOTAL_VRAM"
        "VCN_ACTIVITY"
        "VCN_BUSY"
        "JPEG_ACTIVITY"
        "JPEG_BUSY"
        "XGMI_ERR"
        "WIDTH"
        "SPEED"
    )

    # Split output into lines
    string(REPLACE "\n" ";" _LINES "${AMD_SMI_OUTPUT}")

    foreach(_LINE IN LISTS _LINES)
        # Check for GPU header (e.g., "GPU: 0")
        if(_LINE MATCHES "^GPU:[ \t]*([0-9]+)")
            # Save previous GPU's metrics if any
            if(_CURRENT_GPU GREATER_EQUAL 0 AND _CURRENT_METRICS)
                list(APPEND _GPU_METRICS "GPU${_CURRENT_GPU}:${_CURRENT_METRICS}")
            endif()

            set(_CURRENT_GPU "${CMAKE_MATCH_1}")
            set(_CURRENT_METRICS "")
            math(EXPR _GPU_COUNT "${_GPU_COUNT} + 1")
        endif()

        # Check for supported metrics
        foreach(_METRIC IN LISTS _METRIC_CATEGORIES)
            # Match metric name followed by colon and capture everything after
            if(_LINE MATCHES "^[ \t]*${_METRIC}:[ \t]*(.+)$")
                set(_VALUE "${CMAKE_MATCH_1}")
                string(STRIP "${_VALUE}" _VALUE)

                # Check if value is valid (not N/A)
                _is_metric_value_valid("${_VALUE}" _IS_VALID)

                if(_IS_VALID)
                    # Add metric if not already present (avoid duplicates)
                    if(_CURRENT_METRICS)
                        string(FIND "${_CURRENT_METRICS}" "${_METRIC}" _FOUND_POS)
                        if(_FOUND_POS EQUAL -1)
                            set(_CURRENT_METRICS "${_CURRENT_METRICS},${_METRIC}")
                        endif()
                    else()
                        set(_CURRENT_METRICS "${_METRIC}")
                    endif()
                endif()
                break()
            endif()
        endforeach()
    endforeach()

    # Save last GPU's metrics
    if(_CURRENT_GPU GREATER_EQUAL 0 AND _CURRENT_METRICS)
        list(APPEND _GPU_METRICS "GPU${_CURRENT_GPU}:${_CURRENT_METRICS}")
    endif()

    set(${GPU_COUNT_VAR} ${_GPU_COUNT} PARENT_SCOPE)
    set(${GPU_METRICS_VAR} "${_GPU_METRICS}" PARENT_SCOPE)

    rocprofiler_systems_message(
        STATUS "Found ${_GPU_COUNT} GPU(s) with metrics"
    )
endfunction()

# ---------------------------------------------------------------------------- #
# Main function: Detect all hardware and update existing JSON with calculated values.
# ---------------------------------------------------------------------------- #
function(DETECT_AND_GENERATE_HARDWARE_METRICS OUTPUT_FILE)
    rocprofiler_systems_message(STATUS "Starting hardware detection and updating JSON...")

    # Verify the JSON file exists
    if(NOT EXISTS "${OUTPUT_FILE}")
        rocprofiler_systems_message(
            FATAL_ERROR
            "JSON file does not exist: ${OUTPUT_FILE}. "
            "Please ensure the file exists before running hardware detection."
        )
    endif()

    # Step 1: Run rocminfo
    detect_rocminfo(_ROCMINFO_OUTPUT)
    if(NOT _ROCMINFO_OUTPUT)
        rocprofiler_systems_message(
            WARNING
            "Could not get rocminfo output. "
            "Ensure rocminfo is installed and accessible. "
            "Set ROCM_PATH or add rocminfo to PATH. Skipping validation rules update."
        )
        return()
    endif()

    parse_rocminfo_cpu_count("${_ROCMINFO_OUTPUT}" _CPU_COUNT _GPU_COUNT_ROCM)

    # Step 2: Run amd-smi metric
    detect_amd_smi_metric(_AMD_SMI_OUTPUT)
    if(NOT _AMD_SMI_OUTPUT)
        rocprofiler_systems_message(
            WARNING
            "Could not get amd-smi metric output. "
            "Ensure amd-smi is installed and accessible. "
            "Set ROCM_PATH or add amd-smi to PATH. Skipping validation rules update."
        )
        return()
    endif()

    parse_amd_smi_metrics("${_AMD_SMI_OUTPUT}" _GPU_COUNT _GPU_METRICS)

    # Initialize counters for each metric type
    set(_COUNT_BUSY 0)
    set(_COUNT_TEMP 0)
    set(_COUNT_POWER 0)
    set(_COUNT_MEM_USAGE 0)
    set(_COUNT_VCN 0)
    set(_COUNT_JPEG 0)

    # Calculate GPU metrics count and count GPUs per metric type
    set(_TOTAL_GPU_METRICS 0)
    foreach(_GPU_ENTRY IN LISTS _GPU_METRICS)
        if(_GPU_ENTRY MATCHES "GPU([0-9]+):(.*)")
            set(_METRICS_STR "${CMAKE_MATCH_2}")
            set(_METRIC_COUNT 0)
            if(_METRICS_STR MATCHES "GFX_ACTIVITY")
                math(EXPR _METRIC_COUNT "${_METRIC_COUNT} + 3")
                math(EXPR _COUNT_BUSY "${_COUNT_BUSY} + 1")
            endif()
            if(_METRICS_STR MATCHES "EDGE|HOTSPOT")
                math(EXPR _METRIC_COUNT "${_METRIC_COUNT} + 1")
                math(EXPR _COUNT_TEMP "${_COUNT_TEMP} + 1")
            endif()
            if(_METRICS_STR MATCHES "SOCKET_POWER")
                math(EXPR _METRIC_COUNT "${_METRIC_COUNT} + 1")
                math(EXPR _COUNT_POWER "${_COUNT_POWER} + 1")
            endif()
            if(_METRICS_STR MATCHES "TOTAL_VRAM")
                math(EXPR _METRIC_COUNT "${_METRIC_COUNT} + 1")
                math(EXPR _COUNT_MEM_USAGE "${_COUNT_MEM_USAGE} + 1")
            endif()
            if(_METRICS_STR MATCHES "VCN_ACTIVITY|VCN_BUSY")
                math(EXPR _COUNT_VCN "${_COUNT_VCN} + 1")
            endif()
            if(_METRICS_STR MATCHES "JPEG_ACTIVITY|JPEG_BUSY")
                math(EXPR _COUNT_JPEG "${_COUNT_JPEG} + 1")
            endif()
            math(EXPR _TOTAL_GPU_METRICS "${_TOTAL_GPU_METRICS} + ${_METRIC_COUNT}")
        endif()
    endforeach()

    # Calculate CPU metrics (4 metrics per CPU)
    # CPU metrics collected: thread_cpu_time, thread_page_fault, process_cpu_time, process_page_fault
    math(EXPR _CPU_METRICS "${_CPU_COUNT} * 4")

    # Calculate totals
    math(EXPR _TOTAL_METRICS "${_TOTAL_GPU_METRICS} + ${_CPU_METRICS}")
    set(_INFO_PMC_MIN_ROWS ${_TOTAL_GPU_METRICS})
    set(_PMC_EVENT_MIN_ROWS ${_TOTAL_METRICS})

    # Read existing JSON file
    file(READ "${OUTPUT_FILE}" _JSON_CONTENT)

    # Update _hardware_summary section
    # Update cpu_count
    string(
        REGEX REPLACE
        "\"cpu_count\":[ \t]*[0-9]+"
        "\"cpu_count\": ${_CPU_COUNT}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update cpu_metrics
    string(
        REGEX REPLACE
        "\"cpu_metrics\":[ \t]*[0-9]+"
        "\"cpu_metrics\": ${_CPU_METRICS}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update gpu_count
    string(
        REGEX REPLACE
        "\"gpu_count\":[ \t]*[0-9]+"
        "\"gpu_count\": ${_GPU_COUNT}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update gpu_metrics
    string(
        REGEX REPLACE
        "\"gpu_metrics\":[ \t]*[0-9]+"
        "\"gpu_metrics\": ${_TOTAL_GPU_METRICS}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update total_metrics
    string(
        REGEX REPLACE
        "\"total_metrics\":[ \t]*[0-9]+"
        "\"total_metrics\": ${_TOTAL_METRICS}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update gpus_with_busy
    string(
        REGEX REPLACE
        "\"gpus_with_busy\":[ \t]*[0-9]+"
        "\"gpus_with_busy\": ${_COUNT_BUSY}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update gpus_with_temp
    string(
        REGEX REPLACE
        "\"gpus_with_temp\":[ \t]*[0-9]+"
        "\"gpus_with_temp\": ${_COUNT_TEMP}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update gpus_with_power
    string(
        REGEX REPLACE
        "\"gpus_with_power\":[ \t]*[0-9]+"
        "\"gpus_with_power\": ${_COUNT_POWER}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update gpus_with_mem_usage
    string(
        REGEX REPLACE
        "\"gpus_with_mem_usage\":[ \t]*[0-9]+"
        "\"gpus_with_mem_usage\": ${_COUNT_MEM_USAGE}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update gpus_with_vcn
    string(
        REGEX REPLACE
        "\"gpus_with_vcn\":[ \t]*[0-9]+"
        "\"gpus_with_vcn\": ${_COUNT_VCN}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update gpus_with_jpeg
    string(
        REGEX REPLACE
        "\"gpus_with_jpeg\":[ \t]*[0-9]+"
        "\"gpus_with_jpeg\": ${_COUNT_JPEG}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update rocpd_info_pmc_ table
    # Update min_rows for info_pmc table (first min_rows in the file)
    string(
        REGEX REPLACE
        "(\"required_tables\":[^\\{]+\\{[^}]*\"min_rows\":[ \t]*)[0-9]+"
        "\\1${_INFO_PMC_MIN_ROWS}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update expected_result in first validation query (info_pmc table)
    string(
        REGEX REPLACE
        "(\"query\":[ \t]*\"SELECT COUNT[^\"]*target_arch = 'GPU'\"[^}]*}[^{]*\\{[^}]*\"expected_result\":[ \t]*)[0-9]+"
        "\\1${_INFO_PMC_MIN_ROWS}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update rocpd_pmc_event_ table
    string(
        REGEX REPLACE
        "\"_comment\":[ \t]*\"min_rows = GPU metrics \\([0-9]+\\) \\+ CPU metrics \\([0-9]+ = [0-9]+ CPUs \\* 4\\)\""
        "\"_comment\": \"min_rows = GPU metrics (${_TOTAL_GPU_METRICS}) + CPU metrics (${_CPU_METRICS} = ${_CPU_COUNT} CPUs * 4)\""
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update min_rows for pmc_event table
    string(
        REGEX REPLACE
        "(\"_comment\": \"min_rows = GPU metrics[^\"]+\",\n[ \t]+)\"min_rows\":[ \t]*[0-9]+"
        "\\1\"min_rows\": ${_PMC_EVENT_MIN_ROWS}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update validation query expected_results
    # Update expected_result for busy query
    string(
        REGEX REPLACE
        "(device_busy_mm'\"[^}]*}[^{]*\\{[^}]*\"expected_result\":[ \t]*)[0-9]+"
        "\\1${_COUNT_BUSY}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update expected_result for temp query
    string(
        REGEX REPLACE
        "(device_temp'\"[^}]*}[^{]*\\{[^}]*\"expected_result\":[ \t]*)[0-9]+"
        "\\1${_COUNT_TEMP}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update expected_result for power query
    string(
        REGEX REPLACE
        "(device_power'\"[^}]*}[^{]*\\{[^}]*\"expected_result\":[ \t]*)[0-9]+"
        "\\1${_COUNT_POWER}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Update expected_result for memory_usage query
    string(
        REGEX REPLACE
        "(device_memory_usage'\"[^}]*}[^{]*\\{[^}]*\"expected_result\":[ \t]*)[0-9]+"
        "\\1${_COUNT_MEM_USAGE}"
        _JSON_CONTENT
        "${_JSON_CONTENT}"
    )

    # Write updated content back to file
    file(WRITE "${OUTPUT_FILE}" "${_JSON_CONTENT}")

    rocprofiler_systems_message(STATUS "Hardware detection complete - Updated existing JSON")
    rocprofiler_systems_message(STATUS "  CPUs: ${_CPU_COUNT}, GPUs: ${_GPU_COUNT}")
    rocprofiler_systems_message(STATUS "  CPU metrics: ${_CPU_METRICS}, GPU metrics: ${_TOTAL_GPU_METRICS}")
    rocprofiler_systems_message(STATUS "  Total metrics (pmc_event min_rows): ${_TOTAL_METRICS}")
    rocprofiler_systems_message(STATUS "  GPUs per metric - busy:${_COUNT_BUSY} temp:${_COUNT_TEMP} power:${_COUNT_POWER} mem:${_COUNT_MEM_USAGE} vcn:${_COUNT_VCN} jpeg:${_COUNT_JPEG}")
    rocprofiler_systems_message(STATUS "  File updated : ${OUTPUT_FILE}")
endfunction()

# -----------------------------------------------------------------------------
# Macro: GEN_AMD_SMI_VALIDATION_RULE
# Main entry point for detecting hardware metrics and generating validation rules
# -----------------------------------------------------------------------------
macro(GEN_AMD_SMI_VALIDATION_RULE JSON_FILES_TO_UPDATE)
    message(STATUS "Detecting hardware metrics for rocprofiler-systems...")

    message(STATUS "Updating validation rules: ${JSON_FILES_TO_UPDATE}")
    detect_and_generate_hardware_metrics("${JSON_FILES_TO_UPDATE}")
endmacro()
