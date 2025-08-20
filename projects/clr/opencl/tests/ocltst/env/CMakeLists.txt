add_executable(ocltst)

target_sources(ocltst PRIVATE
    ${OCLTST_DIR}/env/oclTestLog.cpp
    ${OCLTST_DIR}/env/oclsysinfo.cpp
    ${OCLTST_DIR}/env/ocltst.cpp
    ${OCLTST_DIR}/env/pfm.cpp
    ${OCLTST_DIR}/env/Timer.cpp
    ${OCLTST_DIR}/module/common/BaseTestImp.cpp
    ${OCLTST_DIR}/module/common/OCLTestImp.cpp
    ${OCLTST_DIR}/module/common/OCLThread.cpp
    ${OCLTST_DIR}/module/common/OCLWrapper.cpp)

# Windows compatibilty logic
if (WIN32)
    target_sources(ocltst PRIVATE
        ${OCLTST_DIR}/env/getopt.cpp
        ${OCLTST_DIR}/env/ServiceCode.cpp
        ${OCLTST_DIR}/env/window.cpp)
endif()


set_target_properties(ocltst PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst)

target_compile_definitions(ocltst
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_COMPILE_DEFINITIONS>)

target_include_directories(ocltst
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_INCLUDE_DIRECTORIES>)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake")
find_package(AMD_ICD)
find_library(AMD_ICD_LIBRARY OpenCL HINTS "${AMD_ICD_LIBRARY_DIR}")
target_link_libraries(ocltst PRIVATE ${AMD_ICD_LIBRARY})

if (NOT WIN32)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  target_link_libraries(ocltst PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
endif()

set_target_properties(ocltst PROPERTIES INSTALL_RPATH "$ORIGIN")

INSTALL(TARGETS ocltst DESTINATION ${OCLTST_INSTALL_DIR} COMPONENT ocltst)

