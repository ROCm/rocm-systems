#
#
#

set(PACKAGE_OUTPUT_DIR
    ${ROCPROFILER_SDK_TESTS_BINARY_DIR}/pytest-packages/rocprofiler_sdk/pc_sampling/stochastic/csv/gfx9/s_instructions
    )

set(PC_SAMPLING_PYTHON_SOURCES
    __init__.py
    branch_instructions.py
    waitcnt.py
    other_instructions.py
    scalar_instructions.py
    internal_instructions.py
    jump_instructions.py
    message_instructions.py
    barrier_instructions.py)

foreach(_FILE ${PC_SAMPLING_PYTHON_SOURCES})
    configure_file(${CMAKE_CURRENT_LIST_DIR}/${_FILE} ${PACKAGE_OUTPUT_DIR}/${_FILE}
                   COPYONLY)
endforeach()
