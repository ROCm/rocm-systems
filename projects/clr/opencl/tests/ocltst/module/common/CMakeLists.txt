set(COMMON_SOURCES
    ${OCLTST_DIR}/module/common/BaseTestImp.cpp
    ${OCLTST_DIR}/module/common/OCLTestImp.cpp
    ${OCLTST_DIR}/module/common/OCLTestListImp.cpp
    ${OCLTST_DIR}/module/common/OCLTestUtils.cpp
    ${OCLTST_DIR}/module/common/OCLThread.cpp
    ${OCLTST_DIR}/module/common/OCLWrapper.cpp
    ${OCLTST_DIR}/module/common/Timer.cpp)

add_library(Common OBJECT ${COMMON_SOURCES})

set_target_properties(Common PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(Common
    PUBLIC
        CL_TARGET_OPENCL_VERSION=220)

if(EMU_ENV)
    target_compile_definitions(Common
        PUBLIC
            EMU_ENV=1)
endif()

target_include_directories(Common
    PUBLIC
        ${OPENCL_ICD_LOADER_HEADERS_DIR}
        ${OCLTST_DIR}/include
        ${OCLTST_DIR}/module/common
        ${OCLTST_DIR}/module/include
        ${PROJECT_SOURCE_DIR}/amdocl) #TODO remove cl_profile_amd.h dependency

