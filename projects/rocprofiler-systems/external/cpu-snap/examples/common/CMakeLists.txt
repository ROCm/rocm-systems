# Synthetic workload generator (workload.hpp), shared by examples and benches.
add_library(cpu_snap_workload INTERFACE)
target_include_directories(cpu_snap_workload INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(cpu_snap_workload INTERFACE cxx_std_20)

# Header-only demo glue (perfetto_proto_writer.hpp + friends). slice_emitter
# lives in the cpu_snap::processing library component; perfetto_proto_writer
# only consumes its emitted_slice type, so link the library instead of
# building slice_emitter.cpp here.
if(TARGET cpu_snap::processing)
    add_library(perfetto_writer_demo INTERFACE)
    target_include_directories(perfetto_writer_demo INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(perfetto_writer_demo INTERFACE cpu_snap::processing cpu_snap_workload)
    target_compile_features(perfetto_writer_demo INTERFACE cxx_std_20)
endif()
