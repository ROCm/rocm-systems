set(TESTS
    OCLAsyncMap
    OCLAsyncTransfer
    OCLAtomicCounter
    OCLBlitKernel
    OCLBufferFromImage
    OCLCPUGuardPages
    OCLCreateBuffer
    OCLCreateContext
    OCLCreateImage
    OCLDeviceAtomic
    OCLDeviceQueries
    OCLDynamic
    OCLDynamicBLines
    OCLGenericAddressSpace
    OCLGetQueueThreadID
    OCLGlobalOffset
    OCLImage2DFromBuffer
    OCLImageCopyPartial
    OCLKernelBinary
    OCLLDS32K
    OCLLinearFilter
    OCLMapCount
    OCLMemDependency
    OCLMemObjs
    OCLMemoryInfo
    OCLMultiQueue
    OCLOfflineCompilation
    OCLP2PBuffer
    OCLPartialWrkgrp
    OCLPerfCounters
    OCLPersistent
    OCLPinnedMemory
    OCLPlatformAtomics
    OCLProgramScopeVariables
    OCLReadWriteImage
    OCLRTQueue
    OCLSDI
    OCLSemaphore
    OCLStablePState
    OCLSVM
    OCLThreadTrace
    OCLUnalignedCopy
    OCLCreatePipe
)

add_library(oclruntime SHARED
    TestList.cpp
    $<TARGET_OBJECTS:Common>)

foreach(TEST ${TESTS})
    target_sources(oclruntime
        PRIVATE
            ${TEST}.cpp)
endforeach()

set_target_properties(oclruntime PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst)

target_compile_definitions(oclruntime
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_COMPILE_DEFINITIONS>)

target_include_directories(oclruntime
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_INCLUDE_DIRECTORIES>)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake")
find_package(AMD_ICD)
find_library(AMD_ICD_LIBRARY OpenCL HINTS "${AMD_ICD_LIBRARY_DIR}")
target_link_libraries(oclruntime PRIVATE ${AMD_ICD_LIBRARY})

if (NOT WIN32)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  target_link_libraries(oclruntime PRIVATE Threads::Threads)
endif()

add_custom_command(
    TARGET oclruntime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/oclruntime.exclude
            ${CMAKE_BINARY_DIR}/tests/ocltst/oclruntime.exclude)

add_custom_target(test.ocltst.oclruntime
    COMMAND
        ${CMAKE_COMMAND} -E env "OCL_ICD_FILENAMES=$<TARGET_FILE:amdocl>"
        $<TARGET_FILE:ocltst> -p 0 -m $<TARGET_FILE:oclruntime> -A oclruntime.exclude
    DEPENDS
        ocltst oclruntime amdocl
    WORKING_DIRECTORY
        ${CMAKE_BINARY_DIR}/tests/ocltst
    USES_TERMINAL)

foreach(TEST ${TESTS})
    add_custom_target(test.ocltst.oclruntime.${TEST}
        COMMAND
            ${CMAKE_COMMAND} -E env "OCL_ICD_FILENAMES=$<TARGET_FILE:amdocl>"
            $<TARGET_FILE:ocltst> -p 0 -m $<TARGET_FILE:oclruntime> -t ${TEST}
        DEPENDS
            ocltst oclruntime amdocl
        WORKING_DIRECTORY
            ${CMAKE_BINARY_DIR}/tests/ocltst
        USES_TERMINAL)
endforeach()

INSTALL(TARGETS oclruntime DESTINATION ${OCLTST_INSTALL_DIR} COMPONENT ocltst)
INSTALL(FILES oclruntime.exclude DESTINATION ${OCLTST_INSTALL_DIR} COMPONENT ocltst)

