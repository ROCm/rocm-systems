option(BUILD_SHARED "Build shared library" ON)
option(BUILD_STATIC "Build static library" ON)
option(BUILD_INFLUXDB "Build InfluxDB support" OFF)

set(ROCM_TIMESYNC_SOURCES
  consumer.cpp
)

if(BUILD_INFLUXDB)
  list(APPEND ROCM_TIMESYNC_SOURCES influx_client.cpp)
endif()

function(rocm_timesync_configure_target target_name)
  target_link_libraries(${target_name}
    PRIVATE rocm-timesync-core-headers
    PUBLIC rocm-timesync-headers
  )

  if(BUILD_INFLUXDB)
    find_package(CURL REQUIRED)
    find_package(nlohmann_json REQUIRED)

    target_link_libraries(${target_name}
      PRIVATE
        CURL::libcurl
        nlohmann_json::nlohmann_json
    )

    target_compile_definitions(${target_name}
      PRIVATE ROCM_TIMESYNC_BUILD_INFLUXDB=1
    )
  endif()
endfunction()

# Shared library
if(BUILD_SHARED)
  add_library(rocm-timesync SHARED ${ROCM_TIMESYNC_SOURCES} $<TARGET_OBJECTS:rocm-timesync-core>)

  rocm_timesync_configure_target(rocm-timesync)

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
if(BUILD_STATIC)
  add_library(rocm-timesync-static STATIC ${ROCM_TIMESYNC_SOURCES} $<TARGET_OBJECTS:rocm-timesync-core>)

  rocm_timesync_configure_target(rocm-timesync-static)

  set_target_properties(rocm-timesync-static PROPERTIES
    OUTPUT_NAME rocm-timesync
  )

  install(
    TARGETS rocm-timesync-static
    EXPORT rocm-timesync-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
