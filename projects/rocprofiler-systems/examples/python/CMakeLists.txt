cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(omnitrace-python)

set(PYTHON_FILES builtin.py external.py source.py)

if(OMNITRACE_INSTALL_EXAMPLES)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
        set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
        foreach(_FILE ${PYTHON_FILES})
            configure_file(${PROJECT_SOURCE_DIR}/${_FILE} ${PROJECT_BINARY_DIR}/${_FILE}
                           @ONLY)
            install(
                PROGRAMS ${PROJECT_BINARY_DIR}/${_FILE}
                DESTINATION bin
                COMPONENT omnitrace-examples)
        endforeach()
    endif()
endif()
