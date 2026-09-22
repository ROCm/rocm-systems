cmake_minimum_required (VERSION 2.6)

project (rdma_test)

find_package(PkgConfig)
pkg_check_modules(DRM REQUIRED libdrm)
pkg_check_modules(DRM_AMDGPU REQUIRED libdrm_amdgpu)
include_directories(${DRM_AMDGPU_INCLUDE_DIRS})

if( DEFINED ENV{LIBHSAKMT_PATH} )
    set ( LIBHSAKMT_PATH $ENV{LIBHSAKMT_PATH} )
    message ( "LIBHSAKMT_PATH environment variable is set" )
else()
    if ( ${ROCM_INSTALL_PATH} )
       set ( ENV{PKG_CONFIG_PATH} ${ROCM_INSTALL_PATH}/lib/pkgconfig )
    else()
       set ( ENV{PKG_CONFIG_PATH} /opt/rocm/lib/pkgconfig )
    endif()

    pkg_check_modules(HSAKMT libhsakmt)

    if( NOT HSAKMT_FOUND )
       set ( LIBHSAKMT_PATH $ENV{OUT_DIR} )
    endif()
endif()

if( DEFINED LIBHSAKMT_PATH )
    set ( HSAKMT_LIBRARY_DIRS ${LIBHSAKMT_PATH}/lib )
    set ( HSAKMT_LIBRARIES hsakmt )
endif()


link_directories(${HSAKMT_LIBRARY_DIRS})

include_directories($ENV{LIBHSAKMT_ROOT}/include)
include_directories(../drv)

add_executable(rdma_test rdma_test.cpp)
target_link_libraries(rdma_test libhsakmt.a dl pthread numa drm drm_amdgpu)
