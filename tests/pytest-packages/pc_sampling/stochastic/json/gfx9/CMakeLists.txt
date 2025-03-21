#
#
#

set(PACKAGE_OUTPUT_DIR
    ${ROCPROFILER_SDK_TESTS_BINARY_DIR}/pytest-packages/rocprofiler_sdk/pc_sampling/stochastic/json/gfx9
    )

set(PC_SAMPLING_PYTHON_SOURCES __init__.py arbiter_state.py s_instructions.py
                               other_instructions.py)

foreach(_FILE ${PC_SAMPLING_PYTHON_SOURCES})
    configure_file(${CMAKE_CURRENT_LIST_DIR}/${_FILE} ${PACKAGE_OUTPUT_DIR}/${_FILE}
                   COPYONLY)
endforeach()

# add_subdirectory(s_instructions)
