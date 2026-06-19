# Helper script to apply tier labels to install-time CTest files
# This script generates CTest-compatible code that works at runtime.
#
# Usage: Pass test names and test_categories.yaml path, generates set_tests_properties calls

# Function to extract all test names from an install CTestTestfile.cmake
function(extract_test_names_from_file install_file output_var)
    file(READ "${install_file}" _content)

    # Match add_test(name ...) syntax
    string(REGEX MATCHALL "add_test\\(([^ ]+) " _matches "${_content}")

    set(_test_names "")
    foreach(_match ${_matches})
        string(REGEX REPLACE "add_test\\(([^ ]+) " "\\1" _test_name "${_match}")
        list(APPEND _test_names "${_test_name}")
    endforeach()

    list(REMOVE_DUPLICATES _test_names)
    set(${output_var} "${_test_names}" PARENT_SCOPE)
endfunction()

# Function to check if a string contains regex metacharacters
function(is_regex_pattern pattern output_var)
    if(pattern MATCHES "[.*+?^${}()|\\[\\]\\\\]")
        set(${output_var} TRUE PARENT_SCOPE)
    else()
        set(${output_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Function to generate tier labels for a category
# quick       -> quick;standard;comprehensive;full
# standard    -> standard;comprehensive;full
# comprehensive -> comprehensive;full
# full        -> full
function(get_tier_labels category output_var)
    set(TIER_ORDER "quick;standard;comprehensive;full")

    if(category STREQUAL "quick")
        set(${output_var} "quick;standard;comprehensive;full" PARENT_SCOPE)
    elseif(category STREQUAL "standard")
        set(${output_var} "standard;comprehensive;full" PARENT_SCOPE)
    elseif(category STREQUAL "comprehensive")
        set(${output_var} "comprehensive;full" PARENT_SCOPE)
    elseif(category STREQUAL "full")
        set(${output_var} "full" PARENT_SCOPE)
    else()
        set(${output_var} "${category}" PARENT_SCOPE)
    endif()
endfunction()

# Main function to apply tier labels to install CTest file
function(apply_install_tier_labels install_file yaml_file)
    if(NOT EXISTS "${install_file}")
        message(FATAL_ERROR "Install file not found: ${install_file}")
    endif()

    if(NOT EXISTS "${yaml_file}")
        message(FATAL_ERROR "YAML file not found: ${yaml_file}")
    endif()

    # Extract all test names from the install file
    extract_test_names_from_file("${install_file}" _all_test_names)
    list(LENGTH _all_test_names _num_tests)
    message(STATUS "Found ${_num_tests} tests in ${install_file}")

    # Find Python3 to parse the YAML
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    # Use Python to parse YAML and output category assignments
    # We'll create a simple Python script inline
    set(PARSE_YAML_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/parse_yaml_for_install.py")
    file(WRITE "${PARSE_YAML_SCRIPT}" "#!/usr/bin/env python3
import yaml
import sys
import re

def is_regex(s):
    return bool(re.search(r'[.*+?^${}()|\\[\\]\\\\\\\\]', s))

def tier_labels(cat):
    tiers = ['quick', 'standard', 'comprehensive', 'full']
    try:
        idx = tiers.index(cat)
        return tiers[idx:]
    except ValueError:
        return [cat]

yaml_file = sys.argv[1]
with open(yaml_file) as f:
    data = yaml.safe_load(f)

categories = data.get('test_categories', {})

# Output format: category|pattern_type|pattern|labels (using , for label separator)
for cat_name, cat_config in categories.items():
    if not isinstance(cat_config, dict):
        cat_config = {'test_patterns': cat_config or [], 'labels': []}

    patterns = cat_config.get('test_patterns', [])
    custom_labels = cat_config.get('labels', [])

    all_labels = tier_labels(cat_name) + custom_labels
    labels_str = ','.join(dict.fromkeys(all_labels))  # deduplicate, use comma

    for pattern in patterns:
        if is_regex(pattern):
            print(f'{cat_name}|regex|{pattern}|{labels_str}')
        else:
            print(f'{cat_name}|exact|{pattern}|{labels_str}')
")

    # Run Python script to get category data
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${PARSE_YAML_SCRIPT} ${yaml_file}
        OUTPUT_VARIABLE _category_data
        RESULT_VARIABLE _result
        ERROR_VARIABLE _error
    )

    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "Failed to parse YAML: ${_error}")
    endif()

    # Build a mapping of test name -> labels
    # Initialize all tests with empty labels
    foreach(_test ${_all_test_names})
        set(_labels_${_test} "")
    endforeach()

    # Process category data
    string(REPLACE "\n" ";" _category_lines "${_category_data}")
    foreach(_line ${_category_lines})
        if(NOT _line)
            continue()
        endif()

        string(REPLACE "|" ";" _parts "${_line}")
        list(LENGTH _parts _num_parts)
        if(NOT _num_parts EQUAL 4)
            # Skip malformed lines
            continue()
        endif()

        list(GET _parts 0 _cat_name)
        list(GET _parts 1 _pattern_type)
        list(GET _parts 2 _pattern)
        list(GET _parts 3 _labels_comma)

        # Convert comma-separated labels to semicolon-separated (CMake list)
        string(REPLACE "," ";" _labels "${_labels_comma}")

        if(_pattern_type STREQUAL "exact")
            # Exact match - check if test exists in our list
            if(_pattern IN_LIST _all_test_names)
                # Append labels
                if(DEFINED _labels_${_pattern})
                    set(_labels_${_pattern} "${_labels_${_pattern}};${_labels}")
                else()
                    set(_labels_${_pattern} "${_labels}")
                endif()
            endif()
        elseif(_pattern_type STREQUAL "regex")
            # Regex match - check all tests
            foreach(_test ${_all_test_names})
                if(_test MATCHES "^${_pattern}$")
                    # Append labels
                    if(DEFINED _labels_${_test})
                        set(_labels_${_test} "${_labels_${_test}};${_labels}")
                    else()
                        set(_labels_${_test} "${_labels}")
                    endif()
                endif()
            endforeach()
        endif()
    endforeach()

    # Now generate set_tests_properties calls and append to install file
    set(_label_code "\n# Tier labels for monorepo CI (generated by ApplyInstallTierLabels.cmake)\n")
    set(_label_code "${_label_code}# Categories: quick, standard, comprehensive, full\n\n")

    set(_labeled_count 0)
    foreach(_test ${_all_test_names})
        if(DEFINED _labels_${_test})
            # Deduplicate labels and remove any empty elements
            set(_label_list "${_labels_${_test}}")
            list(REMOVE_ITEM _label_list "")
            list(REMOVE_DUPLICATES _label_list)
            list(JOIN _label_list ";" _labels_str)

            if(_labels_str)
                set(_label_code "${_label_code}set_tests_properties(${_test} PROPERTIES LABELS \"${_labels_str}\")\n")
                math(EXPR _labeled_count "${_labeled_count} + 1")
            endif()
        endif()
    endforeach()

    # Append to install file
    file(APPEND "${install_file}" "${_label_code}")

    message(STATUS "Applied tier labels to ${_labeled_count}/${_num_tests} tests in ${install_file}")
endfunction()
