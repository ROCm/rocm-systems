cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(omnitrace-code-coverage LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")
string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")

find_package(Threads REQUIRED)
add_executable(code-coverage code-coverage.cpp)
target_link_libraries(code-coverage PRIVATE Threads::Threads)
target_compile_options(code-coverage PRIVATE ${_FLAGS})

if(OMNITRACE_INSTALL_EXAMPLES)
    install(
        TARGETS code-coverage
        DESTINATION bin
        COMPONENT omnitrace-examples)
endif()

set(PYTHON_FILES code-coverage.py)

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
