option(ROCM_TIMESYNC_BUILD_SHARED "Build shared library" ON)
option(ROCM_TIMESYNC_BUILD_STATIC "Build static library" ON)

set(ROCM_TIMESYNC_SOURCES
  rocm_timesync.cpp
)

# Shared library
if(ROCM_TIMESYNC_BUILD_SHARED)
  add_library(rocm-timesync SHARED ${ROCM_TIMESYNC_SOURCES})

  target_link_libraries(rocm-timesync
    PUBLIC rocm-timesync-headers
  )

  set_target_properties(rocm-timesync PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
  )

  install(
    TARGETS rocm-timesync
    EXPORT rocm-timesync-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

# Static library
if(ROCM_TIMESYNC_BUILD_STATIC)
  add_library(rocm-timesync-static STATIC ${ROCM_TIMESYNC_SOURCES})

  target_link_libraries(rocm-timesync-static
    PUBLIC rocm-timesync-headers
  )

  set_target_properties(rocm-timesync-static PROPERTIES
    OUTPUT_NAME rocm-timesync
  )

  install(
    TARGETS rocm-timesync-static
    EXPORT rocm-timesync-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
