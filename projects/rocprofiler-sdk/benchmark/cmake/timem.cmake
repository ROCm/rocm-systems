#
# timem installation
#

if(NOT ROCPROFILER_BENCHMARK_INSTALL_TIMEM)
    find_program(
        TIMEM_EXECUTABLE
        NAMES timem
        HINTS ${PROJECT_BINARY_DIR}
        PATHS ${PROJECT_BINARY_DIR}
        PATH_SUFFIXES bin)
endif()

if(NOT TIMEM_EXECUTABLE OR NOT EXISTS "${TIMEM_EXECUTABLE}")
    set(TIMEM_INSTALLER
        ${CMAKE_CURRENT_BINARY_DIR}/installer/timemory-timem-1.0.0-Linux.sh)
    find_program(SHELL_EXECUTABLE NAMES sh bash REQUIRED)

    file(
        DOWNLOAD
        https://github.com/ROCm/timemory/releases/download/timemory-timem%2Fv0.0.4/timemory-timem-1.0.0-Linux.sh
        ${TIMEM_INSTALLER}
        EXPECTED_MD5 63da7df7996a86d6d9ce312276c2f014
        INACTIVITY_TIMEOUT 30
        TIMEOUT 300
        SHOW_PROGRESS)

    execute_process(
        COMMAND ${SHELL_EXECUTABLE} ${TIMEM_INSTALLER} --prefix=${PROJECT_BINARY_DIR}
                --exclude-subdir --skip-license
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        RESULT_VARIABLE _RET
        OUTPUT_VARIABLE _OUT
        ERROR_VARIABLE _ERR
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)

    if(NOT EXISTS ${PROJECT_BINARY_DIR}/bin/timem OR NOT _RET EQUAL 0)
        message(
            FATAL_ERROR
                "timem installation failed with exit code ${_RET}.\nSTDOUT:\n\t${_OUT}\nSTDERR:\n\t${_ERR}"
            )
    endif()
endif()

find_program(
    TIMEM_EXECUTABLE
    NAMES timem REQUIRED
    HINTS ${PROJECT_BINARY_DIR}
    PATHS ${PROJECT_BINARY_DIR}
    PATH_SUFFIXES bin)

add_executable(rocprofiler-sdk::timem IMPORTED)
set_property(TARGET rocprofiler-sdk::timem PROPERTY IMPORTED_LOCATION ${TIMEM_EXECUTABLE})
