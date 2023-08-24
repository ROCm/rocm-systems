#
#
#
project(rocprofiler-samples-pc-sampling LANGUAGES C CXX)

add_executable(pc_sampling_single-user-host-trap)
target_sources(pc_sampling_single-user-host-trap PRIVATE common.h
                                                         single-user-host-trap.cpp)
target_link_libraries(pc_sampling_single-user-host-trap
                      PRIVATE rocprofiler::rocprofiler-library)

add_executable(pc_sampling_single-user-host-trap-retry)
target_sources(pc_sampling_single-user-host-trap-retry
               PRIVATE common.h single-user-host-trap-retries-service-instantiation.cpp)
target_link_libraries(pc_sampling_single-user-host-trap-retry
                      PRIVATE rocprofiler::rocprofiler-library)
