cmake_minimum_required (VERSION 2.6)

project (kmtreopen)

link_directories($ENV{ROOT_OF_ROOTS}/out/lib)

include_directories($ENV{LIBHSAKMT_ROOT}/include)

add_executable(kmtreopen kmtreopen.c)
target_link_libraries(kmtreopen libdl.so)

