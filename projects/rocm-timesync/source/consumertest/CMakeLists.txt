set(TEST_SOURCES
    main.cpp
)

add_executable(test
    ${TEST_SOURCES}
    $<TARGET_OBJECTS:rocm-timesync-core>
)

find_package(yaml-cpp REQUIRED)
target_link_libraries(test
    PRIVATE
        rocm-timesync-core-headers 
)

install(
    TARGETS test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
