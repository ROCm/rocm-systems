cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

foreach(BINARY_OUTPUT ${BINARY_DIR}/parser.h ${BINARY_DIR}/parser.cpp
                      ${BINARY_DIR}/scanner.cpp)
    string(REPLACE "${BINARY_DIR}" "${SOURCE_DIR}" SOURCE_OUTPUT "${BINARY_OUTPUT}")
    foreach(VAR PROJECT_SRC_DIR PROJECT_BLD_DIR)
        string(REPLACE "/" "_" ${VAR} "${${VAR}}")
        string(REPLACE "-" "_" ${VAR} "${${VAR}}")
        string(REPLACE "+" "" ${VAR} "${${VAR}}")
        string(TOUPPER "${${VAR}}" ${VAR})
    endforeach()

    # remove absolute path from file
    if(NOT SOURCE_OUTPUT STREQUAL BINARY_OUTPUT)
        file(READ ${BINARY_OUTPUT} OUTPUT_DATA)
        string(REPLACE "${SOURCE_DIR}/" "" OUTPUT_DATA "${OUTPUT_DATA}")
        string(REPLACE "${BINARY_DIR}/" "" OUTPUT_DATA "${OUTPUT_DATA}")
        string(REPLACE "${PROJECT_BLD_DIR}" "_ROCPROFILER" OUTPUT_DATA "${OUTPUT_DATA}")
        string(REPLACE "${PROJECT_SRC_DIR}" "_ROCPROFILER" OUTPUT_DATA "${OUTPUT_DATA}")
        file(WRITE ${BINARY_OUTPUT} "${OUTPUT_DATA}")

        if(FORMAT_EXE)
            execute_process(COMMAND ${FORMAT_EXE} -i ${BINARY_OUTPUT})
        endif()

        configure_file(${BINARY_OUTPUT} ${SOURCE_OUTPUT} COPYONLY)
    endif()
endforeach()
