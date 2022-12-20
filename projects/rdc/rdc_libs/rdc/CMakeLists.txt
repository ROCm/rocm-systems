message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                       Cmake RDC Lib-RDC                           ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INC_DIR "${RDC_LIB_INC_DIR}")

set(RDC_LIB_COMPONENT "lib${RDC_LIB}")
set(RDC_LIB_SRC_LIST ${RDC_LIB_SRC_LIST}
    "${COMMON_DIR}/rdc_capabilities.cc"
    "${COMMON_DIR}/rdc_fields_supported.cc"
    "${SRC_DIR}/RdcCacheManagerImpl.cc"
    "${SRC_DIR}/RdcDiagnosticModule.cc"
    "${SRC_DIR}/RdcEmbeddedHandler.cc"
    "${SRC_DIR}/RdcGroupSettingsImpl.cc"
    "${SRC_DIR}/RdcMetricFetcherImpl.cc"
    "${SRC_DIR}/RdcMetricsUpdaterImpl.cc"
    "${SRC_DIR}/RdcModuleMgrImpl.cc"
    "${SRC_DIR}/RdcNotificationImpl.cc"
    "${SRC_DIR}/RdcPerfTimer.cc"
    "${SRC_DIR}/RdcRasLib.cc"
    "${SRC_DIR}/RdcRocpLib.cc"
    "${SRC_DIR}/RdcRocrLib.cc"
    "${SRC_DIR}/RdcSmiDiagnosticImpl.cc"
    "${SRC_DIR}/RdcSmiLib.cc"
    "${SRC_DIR}/RdcTelemetryModule.cc"
    "${SRC_DIR}/RdcWatchTableImpl.cc"
    "${SRC_DIR}/RsmiUtils.cc")

# TODO: remove all headers? Will just dir be ok after install?
set(RDC_LIB_INC_LIST ${RDC_LIB_INC_LIST}
    "${COMMON_DIR}/rdc_capabilities.h"
    "${COMMON_DIR}/rdc_fields_supported.h"
    "${INC_DIR}/RdcCacheManager.h"
    "${INC_DIR}/RdcDiagnostic.h"
    "${INC_DIR}/RdcDiagnosticLibInterface.h"
    "${INC_DIR}/RdcGroupSettings.h"
    "${INC_DIR}/RdcMetricFetcher.h"
    "${INC_DIR}/RdcMetricsUpdater.h"
    "${INC_DIR}/RdcModuleMgr.h"
    "${INC_DIR}/RdcNotification.h"
    "${INC_DIR}/RdcPerfTimer.h"
    "${INC_DIR}/RdcTelemetry.h"
    "${INC_DIR}/RdcWatchTable.h"
    "${INC_DIR}/impl/RdcCacheManagerImpl.h"
    "${INC_DIR}/impl/RdcDiagnosticModule.h"
    "${INC_DIR}/impl/RdcEmbeddedHandler.h"
    "${INC_DIR}/impl/RdcGroupSettingsImpl.h"
    "${INC_DIR}/impl/RdcMetricFetcherImpl.h"
    "${INC_DIR}/impl/RdcMetricsUpdaterImpl.h"
    "${INC_DIR}/impl/RdcModuleMgrImpl.h"
    "${INC_DIR}/impl/RdcNotificationImpl.h"
    "${INC_DIR}/impl/RdcRasLib.h"
    "${INC_DIR}/impl/RdcRocpLib.h"
    "${INC_DIR}/impl/RdcRocrLib.h"
    "${INC_DIR}/impl/RdcSmiDiagnosticImpl.h"
    "${INC_DIR}/impl/RdcSmiLib.h"
    "${INC_DIR}/impl/RdcTelemetryModule.h"
    "${INC_DIR}/impl/RdcWatchTableImpl.h"
    "${INC_DIR}/impl/RsmiUtils.h")

message("RDC_LIB_INC_LIST=${RDC_LIB_INC_LIST}")

add_library(${RDC_LIB} SHARED ${RDC_LIB_SRC_LIST} ${RDC_LIB_INC_LIST})
target_link_libraries(${RDC_LIB} ${BOOTSTRAP_LIB} pthread rocm_smi64 cap)
target_include_directories(${RDC_LIB} PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/include"
    "${RSMI_INC_DIR}")

# Set the VERSION and SOVERSION values
set_property(TARGET ${RDC_LIB} PROPERTY
    SOVERSION "${VERSION_MAJOR}")
set_property(TARGET ${RDC_LIB} PROPERTY
    VERSION "${SO_VERSION_STRING}")

# If the library is a release, strip the target library
if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    add_custom_command(
        TARGET ${RDC_LIB}
        POST_BUILD COMMAND ${CMAKE_STRIP} ${RDC_LIB_COMPONENT}.so)
endif()
