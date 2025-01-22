if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "Please define SOURCE_DIR")
endif()

get_filename_component(SOURCE_DIR "${SOURCE_DIR}" ABSOLUTE)

find_program(DOT_EXECUTABLE NAMES dot)

if(NOT DOT_EXECUTABLE)
    message(FATAL_ERROR "Please install dot and/or specify DOT_EXECUTABLE")
endif()

file(READ "${SOURCE_DIR}/VERSION" FULL_VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")
string(REGEX REPLACE "([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*)" "\\1.\\2.\\3"
                     ROCPROFILER_VERSION "${FULL_VERSION_STRING}")

configure_file(${SOURCE_DIR}/source/docs/rocprofiler-sdk.dox.in
               ${SOURCE_DIR}/source/docs/rocprofiler-sdk.dox @ONLY)

configure_file(${SOURCE_DIR}/source/docs/rocprofiler-sdk-roctx.dox.in
               ${SOURCE_DIR}/source/docs/rocprofiler-sdk-roctx.dox @ONLY)
