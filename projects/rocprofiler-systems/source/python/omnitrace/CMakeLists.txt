# ########################################################################################
#
# omnitrace (Python)
#
# ########################################################################################

file(GLOB_RECURSE PYTHON_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
foreach(_IN ${PYTHON_FILES})
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}/python/omnitrace"
                   _OUT "${_IN}")
    configure_file(${_IN} ${_OUT} @ONLY)
    install(
        FILES ${_OUT}
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/omnitrace
        OPTIONAL)
endforeach()

# ----------------------------------------------------------------------------
# Console scripts
#
function(OMNITRACE_PYTHON_CONSOLE_SCRIPT SCRIPT_NAME SCRIPT_SUBMODULE)

    configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/console-script.in
                   ${PROJECT_BINARY_DIR}/bin/${SCRIPT_NAME} @ONLY)

    if(CMAKE_INSTALL_PYTHONDIR)
        install(
            PROGRAMS ${PROJECT_BINARY_DIR}/bin/${SCRIPT_NAME}
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            OPTIONAL)
    endif()

    if(OMNITRACE_BUILD_TESTING OR OMNITRACE_BUILD_PYTHON)
        add_test(
            NAME ${SCRIPT_NAME}-console-script-test
            COMMAND ${PROJECT_BINARY_DIR}/bin/${SCRIPT_NAME} --help
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        set_tests_properties(
            ${SCRIPT_NAME}-console-script-test
            PROPERTIES ENVIRONMENT
                       "PYTHONPATH=${PROJECT_BINARY_DIR}/python:$ENV{PYTHONPATH}")
    endif()
endfunction()

if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "python${OMNITRACE_PYTHON_VERSION}")
endif()

if(OMNITRACE_USE_PYTHON)
    omnitrace_python_console_script("omnitrace-python" "omnitrace")
endif()
