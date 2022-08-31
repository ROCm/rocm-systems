cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(omnitrace-python)

set(PYTHON_FILES builtin.py external.py source.py noprofile.py fill.py)

find_package(Python3 COMPONENTS Interpreter)

if(Python3_FOUND)
    set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")

    foreach(_FILE ${PYTHON_FILES})
        configure_file(${PROJECT_SOURCE_DIR}/${_FILE} ${PROJECT_BINARY_DIR}/${_FILE}
                       @ONLY)

        if(OMNITRACE_INSTALL_EXAMPLES)
            install(
                PROGRAMS ${PROJECT_BINARY_DIR}/${_FILE}
                DESTINATION bin
                COMPONENT omnitrace-examples)
        endif()
    endforeach()
endif()
