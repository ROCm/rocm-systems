add_library(core STATIC
    module_map.cpp
    reader.cpp
    sample_reconstructor.cpp)
add_library(cpu_snap::core ALIAS core)

target_include_directories(core PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(core PUBLIC Threads::Threads)
target_compile_features(core PUBLIC cxx_std_20)
target_compile_options(core PRIVATE
    -Wall -Wextra -Wpedantic -fno-omit-frame-pointer)

if(CPU_SNAP_INSTALL)
    install(TARGETS core
        EXPORT  cpu_snapTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${PROJECT_SOURCE_DIR}/src/cpu_snap/error.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cpu_snap)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cpu_snap
        FILES_MATCHING PATTERN "*.hpp")
endif()
