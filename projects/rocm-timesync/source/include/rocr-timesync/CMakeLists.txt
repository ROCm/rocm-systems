add_library(rocr-timesync-headers INTERFACE)

target_include_directories(rocr-timesync-headers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocr-timesync
)

install(
  TARGETS rocr-timesync-headers
  EXPORT rocr-timesync-targets
)
