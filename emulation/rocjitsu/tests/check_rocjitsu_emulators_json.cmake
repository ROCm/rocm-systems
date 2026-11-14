if(NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()

execute_process(
    COMMAND "${TEST_EXECUTABLE}" emulators --json
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
)

if(NOT _result EQUAL 0)
    message(
        FATAL_ERROR
        "rocjitsu emulators --json failed (${_result}): ${_stderr}"
    )
endif()

string(JSON _name GET "${_stdout}" 0 name)
if(NOT _name STREQUAL "rocjitsu")
    message(FATAL_ERROR "expected first emulator name rocjitsu, got '${_name}'")
endif()

string(JSON _default GET "${_stdout}" 0 default)
if(NOT _default)
    message(FATAL_ERROR "expected rocjitsu emulator to be the default")
endif()

string(JSON _runtime_state GET "${_stdout}" 0 runtime state)
if(_runtime_state STREQUAL "")
    message(FATAL_ERROR "expected runtime.state in emulator JSON")
endif()

string(JSON _options_type TYPE "${_stdout}" 0 options)
if(NOT _options_type STREQUAL "ARRAY")
    message(FATAL_ERROR "expected options to be an array, got ${_options_type}")
endif()
