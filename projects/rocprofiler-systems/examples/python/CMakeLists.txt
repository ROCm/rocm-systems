cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

project(rocprofiler-systems-python)

if(ROCPROFSYS_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(
        ${PROJECT_NAME} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
        OR ${_DIR} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
    )
        set(ROCPROFSYS_DISABLE_PYTHON_CTEST ON CACHE BOOL "Disable Python CTests")
        return()
    endif()
endif()

set(PYTHON_FILES
    builtin.py
    external.py
    source.py
    noprofile.py
    fill.py
)

find_package(Python3 COMPONENTS Interpreter)

if(Python3_FOUND)
    set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")

    foreach(_FILE ${PYTHON_FILES})
        configure_file(
            ${PROJECT_SOURCE_DIR}/${_FILE}
            ${PROJECT_BINARY_DIR}/${_FILE}
            @ONLY
        )

        if(ROCPROFSYS_INSTALL_EXAMPLES)
            install(
                PROGRAMS ${PROJECT_BINARY_DIR}/${_FILE}
                DESTINATION bin
                COMPONENT rocprofiler-systems-examples
            )
        endif()
    endforeach()
endif()
