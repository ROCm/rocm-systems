# Offline, post-capture processing: symbol resolution (cpu_snap::processing::symbols)
# plus timeline shaping (cpu_snap::processing::timeline). No libdw, not capture-time.
add_library(processing STATIC timeline/slice_emitter.cpp)
add_library(cpu_snap::processing ALIAS processing)

target_include_directories(processing PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(processing PUBLIC cpu_snap::core)
target_compile_features(processing PUBLIC cxx_std_20)
target_compile_options(processing PRIVATE
    -Wall -Wextra -Wpedantic -fno-omit-frame-pointer)

if(CPU_SNAP_INSTALL)
    install(TARGETS processing
        EXPORT  cpu_snapTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cpu_snap
        FILES_MATCHING PATTERN "*.hpp")
endif()
