cmake_minimum_required(VERSION 3.22)

project(
  rocr-timesync
  VERSION 0.1.0
  LANGUAGES CXX
)

include(GNUInstallDirs)
set(CMAKE_INSTALL_LIBDIR lib)

add_subdirectory(source)

install(
  EXPORT rocr-timesync-targets
  NAMESPACE rocr-timesync::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocr-timesync
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocr-timesync-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/rocr-timesync-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocr-timesync
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/rocr-timesync-config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocr-timesync
)
