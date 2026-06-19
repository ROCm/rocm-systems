add_library(rocm-timesync-core-headers INTERFACE)

target_include_directories(rocm-timesync-core-headers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

install(
  TARGETS rocm-timesync-core-headers
  EXPORT rocm-timesync-targets
)
