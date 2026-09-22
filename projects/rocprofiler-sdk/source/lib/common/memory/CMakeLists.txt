#
# add container sources and headers to common library target
#
set(memory_headers deleter.hpp pool.hpp pool_allocator.hpp stateless_allocator.hpp)
set(memory_sources)

target_sources(rocprofiler-sdk-common-library PRIVATE ${memory_sources} ${memory_headers})
