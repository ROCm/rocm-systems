option(ROCR_TIMESYNC_BUILD_SHARED "Build shared library" ON)
option(ROCR_TIMESYNC_BUILD_STATIC "Build static library" ON)

set(ROCR_TIMESYNC_SOURCES
  rocr_timesync.cpp
)

# Shared library
if(ROCR_TIMESYNC_BUILD_SHARED)
  add_library(rocr-timesync SHARED ${ROCR_TIMESYNC_SOURCES})

  target_link_libraries(rocr-timesync
    PUBLIC rocr-timesync-headers
  )

  set_target_properties(rocr-timesync PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
  )

  install(
    TARGETS rocr-timesync
    EXPORT rocr-timesync-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

# Static library
if(ROCR_TIMESYNC_BUILD_STATIC)
  add_library(rocr-timesync-static STATIC ${ROCR_TIMESYNC_SOURCES})

  target_link_libraries(rocr-timesync-static
    PUBLIC rocr-timesync-headers
  )

  set_target_properties(rocr-timesync-static PROPERTIES
    OUTPUT_NAME rocr-timesync
  )

  install(
    TARGETS rocr-timesync-static
    EXPORT rocr-timesync-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
