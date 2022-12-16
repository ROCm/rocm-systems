cmake_minimum_required(VERSION 3.8)

if(NOT DEFINED OMNITRACE_VERSION)
    file(READ "${CMAKE_CURRENT_LIST_DIR}/../VERSION" FULL_VERSION_STRING LIMIT_COUNT 1)
    string(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")
    string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\1.\\2.\\3"
                         OMNITRACE_VERSION "${FULL_VERSION_STRING}")
endif()

if(NOT DEFINED OUTPUT_DIR)
    set(OUTPUT_DIR ${CMAKE_CURRENT_LIST_DIR})
endif()

message(
    STATUS
        "Writing ${OUTPUT_DIR}/omnitrace-install.py for omnitrace v${OMNITRACE_VERSION}")

configure_file(${CMAKE_CURRENT_LIST_DIR}/../cmake/Templates/omnitrace-install.py.in
               ${OUTPUT_DIR}/omnitrace-install.py @ONLY)
