set(TESTS
    OCLGLBuffer
    OCLGLBufferMultipleQueues
    OCLGLDepthBuffer
    OCLGLDepthTex
    OCLGLFenceSync
    OCLGLMsaaTexture
    OCLGLMultiContext
    OCLGLTexture
    OCLGLPerfSepia
)

add_library(oclgl SHARED
    TestList.cpp
    $<TARGET_OBJECTS:Common>)

foreach(TEST ${TESTS})
    target_sources(oclgl
        PRIVATE
            ${TEST}.cpp)
endforeach()

target_sources(oclgl
    PRIVATE
        ${OCLTST_DIR}/module/common/OCLGLCommon.cpp
        ${OCLTST_DIR}/module/common/OCLGLCommonLinux.cpp)

target_include_directories(oclgl
    PUBLIC
        ${OPENGL_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIRS})

set_target_properties(oclgl PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst)

target_compile_definitions(oclgl
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_COMPILE_DEFINITIONS>)

target_include_directories(oclgl
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(oclgl
    PRIVATE
        GLEW::GLEW
        ${OPENGL_LIBRARIES})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake")
find_package(AMD_ICD)
find_library(AMD_ICD_LIBRARY OpenCL HINTS "${AMD_ICD_LIBRARY_DIR}")
target_link_libraries(oclgl PRIVATE ${AMD_ICD_LIBRARY})

if (NOT WIN32)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  target_link_libraries(oclgl PRIVATE Threads::Threads)
endif()

add_custom_command(
    TARGET oclgl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/oclgl.exclude
            ${CMAKE_BINARY_DIR}/tests/ocltst/ogl.exclude)

add_custom_target(test.ocltst.oclgl
    COMMAND
        ${CMAKE_COMMAND} -E env "OCL_ICD_FILENAMES=$<TARGET_FILE:amdocl>"
        $<TARGET_FILE:ocltst> -p 0 -m $<TARGET_FILE:oclgl> -A oclgl.exclude
    DEPENDS
        ocltst oclgl amdocl
    WORKING_DIRECTORY
        ${CMAKE_BINARY_DIR}/tests/ocltst
    USES_TERMINAL)

foreach(TEST ${TESTS})
    add_custom_target(test.ocltst.oclgl.${TEST}
        COMMAND
            ${CMAKE_COMMAND} -E env "OCL_ICD_FILENAMES=$<TARGET_FILE:amdocl>"
            $<TARGET_FILE:ocltst> -p 0 -m $<TARGET_FILE:oclgl> -t ${TEST}
        DEPENDS
            ocltst oclgl amdocl
        WORKING_DIRECTORY
            ${CMAKE_BINARY_DIR}/tests/ocltst
        USES_TERMINAL)
endforeach()

INSTALL(TARGETS oclgl DESTINATION ${OCLTST_INSTALL_DIR} COMPONENT ocltst)
INSTALL(FILES oclgl.exclude DESTINATION ${OCLTST_INSTALL_DIR} COMPONENT ocltst)

