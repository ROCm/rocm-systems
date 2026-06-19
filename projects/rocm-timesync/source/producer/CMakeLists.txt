set(PRODUCER_SOURCES
    producer.cpp
    main.cpp
    config.cpp
)

add_executable(rocm-timesyncd
    ${PRODUCER_SOURCES}
    $<TARGET_OBJECTS:rocm-timesync-core>
)

find_package(yaml-cpp REQUIRED)
target_link_libraries(rocm-timesyncd
    PRIVATE
        rocm-timesync-core-headers 
        yaml-cpp
)

install(
    TARGETS rocm-timesyncd
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
