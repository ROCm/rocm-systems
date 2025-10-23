cmake_minimum_required(VERSION 3.10)
project(cuid_example)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenSSL REQUIRED)

add_executable(cuid_example main.cc)

target_include_directories(cuid_example PRIVATE ${CMAKE_SOURCE_DIR}/../lib)
target_link_libraries(cuid_example PRIVATE cuid_static OpenSSL::Crypto pthread)
