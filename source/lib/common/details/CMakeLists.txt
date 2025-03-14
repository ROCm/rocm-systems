#
# add details sources and headers to common library target
#
set(details_headers mpl.hpp)
set(details_sources)

target_sources(rocprofiler-sdk-common-library PRIVATE ${details_sources}
                                                      ${details_headers})
