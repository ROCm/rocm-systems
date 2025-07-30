cmake_minimum_required(VERSION 3.20)

# Generate Unified Header amdsmi.h
if(BUILD_HEADER)
    set(UNIFIED_DIR "unified_amdsmi")
    set(UNIFIED_H "${UNIFIED_DIR}/amdsmi.h")
    set(AMDSMI_H "amd_smi/amdsmi.h")
    execute_process(
        COMMAND ./create_header.py linguest_gpu_cpu
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${UNIFIED_DIR}/scripts
        OUTPUT_VARIABLE pyout
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
        COMMAND mv -f ${UNIFIED_H} ${AMDSMI_H}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE lsout
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
