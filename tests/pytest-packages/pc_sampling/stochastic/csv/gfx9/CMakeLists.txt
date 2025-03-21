#
#
#

set(PACKAGE_OUTPUT_DIR
    ${ROCPROFILER_SDK_TESTS_BINARY_DIR}/pytest-packages/rocprofiler_sdk/pc_sampling/stochastic/csv/gfx9
    )

set(PC_SAMPLING_PYTHON_SOURCES
    __init__.py valu_instructions.py matrix_instructions.py texture_instructions.py
    flat_instructions.py lds_instructions.py)

foreach(_FILE ${PC_SAMPLING_PYTHON_SOURCES})
    configure_file(${CMAKE_CURRENT_LIST_DIR}/${_FILE} ${PACKAGE_OUTPUT_DIR}/${_FILE}
                   COPYONLY)
endforeach()

add_subdirectory(s_instructions)
