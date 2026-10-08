set(PRODUCER_SOURCES
    main.cpp
    config.cpp
    kfd.cpp
)

add_executable(rocm-timesyncd
    ${PRODUCER_SOURCES}
    $<TARGET_OBJECTS:rocm-timesync-core>
)

# to avoid a circular dependency between ourselves and rocr-runtime
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../rocr-runtime/libhsakmt
    ${CMAKE_BINARY_DIR}/hsakmt
    EXCLUDE_FROM_ALL
)
#find_package(hsakmt REQUIRED)

find_package(yaml-cpp REQUIRED)

target_link_libraries(rocm-timesyncd
    PRIVATE
        rocm-timesync-core-headers
        hsakmt::hsakmt
        yaml-cpp
)

install(
    TARGETS rocm-timesyncd
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
