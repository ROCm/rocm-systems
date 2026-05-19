# This script reads the test categories YAML file and applies labels to CTest

# Find Python3 for running the parser script
find_package(Python3 COMPONENTS Interpreter)

# Function to apply category labels to discovered GTest tests
# Optional 2nd parameter: install_test_file - path to write install-time test definitions
function(apply_ctest_category_labels yaml_file)
    # Execute the Python script to generate CMake code
    if(NOT Python3_FOUND)
        message(WARNING "Python3 not found, cannot parse test categories YAML")
        return()
    endif()

    # Validate inputs
    set(_validation_failed FALSE)
    if(NOT EXISTS "${yaml_file}")
        message(WARNING "Test categories YAML file not found: ${yaml_file}")
        set(_validation_failed TRUE)
    endif()
    if(_validation_failed)
        return()
    endif()

    # Verify the parser script exists
    set(PARSE_SCRIPT
        "${ROCM_SYSTEMS_ROOT}/shared/ctest/parse_ctest_categories.py"
    )
    if(NOT EXISTS "${PARSE_SCRIPT}")
        message(
            WARNING
            "Test category parser script not found: ${PARSE_SCRIPT}"
        )
        return()
    endif()

    # Check if optional install_test_file parameter was provided
    set(install_test_file "${ARGV1}")
    if(install_test_file)
        set(python_args ${yaml_file} ${install_test_file})
    else()
        set(python_args ${yaml_file})
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${PARSE_SCRIPT} ${python_args}
        OUTPUT_VARIABLE CMAKE_CATEGORY_CODE
        ERROR_VARIABLE PARSE_ERROR
        RESULT_VARIABLE PARSE_RESULT
    )

    if(NOT PARSE_RESULT EQUAL 0)
        message(WARNING "Failed to parse test categories YAML: ${PARSE_ERROR}")
        return()
    endif()

    # Write the generated CMake code to a file and include it
    set(CATEGORY_CMAKE "${CMAKE_CURRENT_BINARY_DIR}/test_categories.cmake")
    file(WRITE "${CATEGORY_CMAKE}" "${CMAKE_CATEGORY_CODE}")

    message(STATUS "Generated test category configuration: ${CATEGORY_CMAKE}")

    # Verify the generated CMake file exists before including it
    if(NOT EXISTS "${CATEGORY_CMAKE}")
        message(
            WARNING
            "Generated test categories file not found: ${CATEGORY_CMAKE}"
        )
        return()
    endif()

    # Include and execute the generated CMake code
    include("${CATEGORY_CMAKE}")
endfunction()

# Function to apply category labels to discovered GTest tests
# Optional 4th parameter: install_test_file - path to write install-time test definitions
function(apply_test_category_labels target_name yaml_file working_dir)
    # Execute the Python script to generate CMake code
    if(NOT Python3_FOUND)
        message(WARNING "Python3 not found, cannot parse test categories YAML")
        return()
    endif()

    # Validate inputs
    set(_validation_failed FALSE)
    if("${target_name}" STREQUAL "")
        message(WARNING "target_name is empty, cannot generate test categories")
        set(_validation_failed TRUE)
    endif()
    if(NOT EXISTS "${yaml_file}")
        message(WARNING "Test categories YAML file not found: ${yaml_file}")
        set(_validation_failed TRUE)
    endif()
    if(NOT IS_DIRECTORY "${working_dir}")
        message(WARNING "Working directory does not exist: ${working_dir}")
        set(_validation_failed TRUE)
    endif()
    if(_validation_failed)
        return()
    endif()

    # Verify the parser script exists
    set(PARSE_SCRIPT
        "${ROCM_SYSTEMS_ROOT}/shared/ctest/parse_test_categories.py"
    )
    if(NOT EXISTS "${PARSE_SCRIPT}")
        message(
            WARNING
            "Test category parser script not found: ${PARSE_SCRIPT}"
        )
        return()
    endif()

    # Check if optional install_test_file parameter was provided
    set(install_test_file "${ARGV3}")
    if(install_test_file)
        set(python_args
            ${yaml_file}
            ${target_name}
            ${working_dir}
            ${install_test_file}
        )
    else()
        set(python_args ${yaml_file} ${target_name} ${working_dir})
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${PARSE_SCRIPT} ${python_args}
        OUTPUT_VARIABLE CMAKE_CATEGORY_CODE
        ERROR_VARIABLE PARSE_ERROR
        RESULT_VARIABLE PARSE_RESULT
    )

    if(NOT PARSE_RESULT EQUAL 0)
        message(WARNING "Failed to parse test categories YAML: ${PARSE_ERROR}")
        return()
    endif()

    # Write the generated CMake code to a file and include it
    set(CATEGORY_CMAKE "${CMAKE_CURRENT_BINARY_DIR}/test_categories.cmake")
    file(WRITE "${CATEGORY_CMAKE}" "${CMAKE_CATEGORY_CODE}")

    message(STATUS "Generated test category configuration: ${CATEGORY_CMAKE}")

    # Verify the generated CMake file exists before including it
    if(NOT EXISTS "${CATEGORY_CMAKE}")
        message(
            WARNING
            "Generated test categories file not found: ${CATEGORY_CMAKE}"
        )
        return()
    endif()

    # Include and execute the generated CMake code
    include("${CATEGORY_CMAKE}")
endfunction()
