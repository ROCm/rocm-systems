add_library(rocm-timesync-headers INTERFACE)

target_include_directories(rocm-timesync-headers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocm-timesync
)

install(
  TARGETS rocm-timesync-headers
  EXPORT rocm-timesync-targets
)
