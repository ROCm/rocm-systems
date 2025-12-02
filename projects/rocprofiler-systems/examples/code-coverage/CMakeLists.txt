cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(rocprofiler-systems-code-coverage-example LANGUAGES CXX)

if(ROCPROFSYS_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(
        ${PROJECT_NAME} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
        OR ${_DIR} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
    )
        return()
    endif()
endif()

set(CMAKE_BUILD_TYPE "RelWithDebInfo")
string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")

find_package(Threads REQUIRED)
add_executable(code-coverage code-coverage.cpp)
target_link_libraries(code-coverage PRIVATE Threads::Threads)
target_compile_options(code-coverage PRIVATE ${_FLAGS})

set_target_properties(
    code-coverage
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/code-coverage"
)

if(ROCPROFSYS_INSTALL_EXAMPLES)
    install(
        TARGETS code-coverage
        DESTINATION share/rocprofiler-systems/samples/code-coverage
        COMPONENT rocprofiler-systems-samples
    )
endif()

set(PYTHON_FILES code-coverage.py)

find_package(Python3 COMPONENTS Interpreter)

if(Python3_FOUND)
    set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")

    foreach(_FILE ${PYTHON_FILES})
        configure_file(
            ${PROJECT_SOURCE_DIR}/${_FILE}
            ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/samples/code-coverage/${_FILE}
            @ONLY
        )

        if(ROCPROFSYS_INSTALL_EXAMPLES)
            install(
                PROGRAMS
                    ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/samples/code-coverage/${_FILE}
                DESTINATION share/rocprofiler-systems/samples/code-coverage
                COMPONENT rocprofiler-systems-samples
            )
        endif()
    endforeach()
endif()
