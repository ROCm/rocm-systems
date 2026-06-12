cmake_minimum_required(VERSION 3.22)

project(
  rocm-timesync
  VERSION 0.1.0
  LANGUAGES CXX
)

include(GNUInstallDirs)
set(CMAKE_INSTALL_LIBDIR lib)

add_subdirectory(source)

install(
  EXPORT rocm-timesync-targets
  NAMESPACE rocm-timesync::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocm-timesync
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocm-timesync-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/rocm-timesync-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocm-timesync
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/rocm-timesync-config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocm-timesync
)
