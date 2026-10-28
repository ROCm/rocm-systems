# cpu-snap-run needs the autoload example, which needs sampling + processing.
if(NOT TARGET callstack_autoload)
    return()
endif()

# cpu-snap-run: run any program under the sampler (preloads libcallstack_autoload).
add_executable(cpu_snap_run cpu_snap_run/main.cpp)
set_target_properties(cpu_snap_run PROPERTIES OUTPUT_NAME cpu-snap-run)
target_include_directories(cpu_snap_run PRIVATE cpu_snap_run)
target_compile_features(cpu_snap_run PRIVATE cxx_std_20)
target_compile_options(cpu_snap_run PRIVATE -Wall -Wextra -Wpedantic)
# Build-tree default for --preload; CPU_SNAP_PRELOAD / --preload override it.
target_compile_definitions(cpu_snap_run PRIVATE
    CPU_SNAP_RUN_DEFAULT_PRELOAD="$<TARGET_FILE:callstack_autoload>")
add_dependencies(cpu_snap_run callstack_autoload)
