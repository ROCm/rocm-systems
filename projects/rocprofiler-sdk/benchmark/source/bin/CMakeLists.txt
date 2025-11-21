#
#
#

# Adding main rocprofv3
configure_file(rocprofv3-benchmark.py
               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocprofv3-benchmark @ONLY)

install(
    FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocprofv3-benchmark
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ
                WORLD_EXECUTE)

# downloaded timem
install(
    FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/timem
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ
                WORLD_EXECUTE
    OPTIONAL)

# Adding Benchmark Workloads
add_subdirectory(mandelbrot)
