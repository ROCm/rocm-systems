set(TESTS
    OCLPerf3DImageWriteSpeed
    OCLPerfAES256
    OCLPerfAtomicSpeed20
    OCLPerfAtomicSpeed
    OCLPerfBufferCopyOverhead
    OCLPerfBufferCopySpeed
    OCLPerfBufferReadSpeed
    OCLPerfBufferWriteSpeed
    OCLPerfCommandQueue
    OCLPerfConcurrency
    OCLPerfCPUMemSpeed
    OCLPerfDeviceConcurrency
    OCLPerfDeviceEnqueue
    OCLPerfDeviceEnqueue2
    OCLPerfDeviceEnqueueEvent
    OCLPerfDeviceEnqueueSier
    OCLPerfDevMemReadSpeed
    OCLPerfDevMemWriteSpeed
    OCLPerfDispatchSpeed
    OCLPerfDoubleDMA
    OCLPerfDoubleDMASeq
    OCLPerfFillBuffer
    OCLPerfFillImage
    OCLPerfFlush
    OCLPerfGenericBandwidth
    OCLPerfGenoilSiaMiner
    OCLPerfImageCopyCorners
    OCLPerfImageCopySpeed
    OCLPerfImageCreate
    OCLPerfImageMapUnmap
    OCLPerfImageReadSpeed
    OCLPerfImageReadsRGBA
    OCLPerfImageReadWrite
    OCLPerfImageSampleRate
    OCLPerfImageWriteSpeed
    OCLPerfKernelArguments
    OCLPerfKernelThroughput
    OCLPerfLDSLatency
    OCLPerfLDSReadSpeed
    OCLPerfMandelbrot
    OCLPerfMapBufferReadSpeed
    OCLPerfMapBufferWriteSpeed
    OCLPerfMapImageReadSpeed
    OCLPerfMapImageWriteSpeed
    OCLPerfMatrixTranspose
    OCLPerfMemCombine
    OCLPerfMemCreate
    OCLPerfMemLatency
    OCLPerfPinnedBufferReadSpeed
    OCLPerfPinnedBufferWriteSpeed
    OCLPerfPipeCopySpeed
    OCLPerfProgramGlobalRead
    OCLPerfProgramGlobalWrite
    OCLPerfSampleRate
    OCLPerfScalarReplArrayElem
    OCLPerfSdiP2PCopy
    OCLPerfSHA256
    OCLPerfSVMAlloc
    OCLPerfSVMKernelArguments
    OCLPerfSVMMap
    OCLPerfSVMMemcpy
    OCLPerfSVMMemFill
    OCLPerfSVMSampleRate
    OCLPerfTextureMemLatency
    OCLPerfUAVReadSpeed
    OCLPerfUAVReadSpeedHostMem
    OCLPerfUAVWriteSpeedHostMem
    OCLPerfUncoalescedRead
    OCLPerfVerticalFetch
)

add_library(oclperf SHARED
    TestList.cpp
    $<TARGET_OBJECTS:Common>)

foreach(TEST ${TESTS})
    target_sources(oclperf
        PRIVATE
            ${TEST}.cpp)
endforeach()

set_target_properties(oclperf PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst)

target_compile_definitions(oclperf
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_COMPILE_DEFINITIONS>)

target_include_directories(oclperf
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_INCLUDE_DIRECTORIES>)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake")
find_package(AMD_ICD)
find_library(AMD_ICD_LIBRARY OpenCL HINTS "${AMD_ICD_LIBRARY_DIR}")
target_link_libraries(oclperf PRIVATE ${AMD_ICD_LIBRARY})
if (NOT WIN32)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  target_link_libraries(oclperf PRIVATE Threads::Threads)
endif()

add_custom_command(
    TARGET oclperf POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/oclperf.exclude
            ${CMAKE_BINARY_DIR}/tests/ocltst/oclperf.exclude)

add_custom_target(test.ocltst.oclperf
    COMMAND
        ${CMAKE_COMMAND} -E env "OCL_ICD_FILENAMES=$<TARGET_FILE:amdocl>"
        $<TARGET_FILE:ocltst> -p 0 -m $<TARGET_FILE:oclperf> -A oclperf.exclude
    DEPENDS
        ocltst oclperf amdocl
    WORKING_DIRECTORY
        ${CMAKE_BINARY_DIR}/tests/ocltst
    USES_TERMINAL)

foreach(TEST ${TESTS})
    add_custom_target(test.ocltst.oclperf.${TEST}
        COMMAND
            ${CMAKE_COMMAND} -E env "OCL_ICD_FILENAMES=$<TARGET_FILE:amdocl>"
            $<TARGET_FILE:ocltst> -p 0 -m $<TARGET_FILE:oclperf> -t ${TEST}
        DEPENDS
            ocltst oclperf amdocl
        WORKING_DIRECTORY
            ${CMAKE_BINARY_DIR}/tests/ocltst
        USES_TERMINAL)
endforeach()

INSTALL(TARGETS oclperf DESTINATION ${OCLTST_INSTALL_DIR} COMPONENT ocltst)
INSTALL(FILES oclperf.exclude DESTINATION ${OCLTST_INSTALL_DIR} COMPONENT ocltst)

