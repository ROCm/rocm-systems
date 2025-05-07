message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                       Cmake RDC Lib-Client                        ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INC_DIR "${RDC_LIB_INC_DIR}")

file(GLOB PROTOBUF_GENERATED_INCLUDES "${PROTOB_OUT_DIR}/*.h")
file(GLOB PROTOBUF_GENERATED_SRCS "${PROTOB_OUT_DIR}/*.cc")

set(RDCCLIENT_LIB_COMPONENT "lib${RDCCLIENT_LIB}")
set(RDCCLIENT_LIB_SRC_LIST "${SRC_DIR}/RdcStandaloneHandler.cc" "${PROTOBUF_GENERATED_SRCS}")

set(RDCCLIENT_LIB_INC_LIST "${PROJECT_SOURCE_DIR}/include/rdc/rdc.h" "${INC_DIR}/rdc_common.h"
                           "${INC_DIR}/RdcHandler.h" "${INC_DIR}/impl/RdcStandaloneHandler.h")

message("RDCCLIENT_LIB_INC_LIST=${RDCCLIENT_LIB_INC_LIST}")

add_library(${RDCCLIENT_LIB} SHARED ${RDCCLIENT_LIB_SRC_LIST} ${RDCCLIENT_LIB_INC_LIST})
target_link_libraries(${RDCCLIENT_LIB} ${BOOTSTRAP_LIB} pthread rt gRPC::grpc++ dl)
target_include_directories(
    ${RDCCLIENT_LIB} PRIVATE "${GRPC_ROOT}/include" "${PROJECT_SOURCE_DIR}"
                             "${PROJECT_SOURCE_DIR}/include" "${PROTOB_OUT_DIR}")

# Set the VERSION and SOVERSION values
set_property(TARGET ${RDCCLIENT_LIB} PROPERTY SOVERSION "${VERSION_MAJOR}")
set_property(TARGET ${RDCCLIENT_LIB} PROPERTY VERSION "${SO_VERSION_STRING}")

if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    add_custom_command(
        TARGET ${RDCCLIENT_LIB}
        POST_BUILD
        COMMAND ${CMAKE_STRIP} ${RDCCLIENT_LIB_COMPONENT}.so)
endif()
