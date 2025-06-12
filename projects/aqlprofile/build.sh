#!/bin/bash -e



SRC_DIR=$(dirname "$0")
COMPONENT="aqlprofile"
ROCM_PATH="${ROCM_PATH:=/opt/rocm}"
LD_RUNPATH_FLAG=" -Wl,--enable-new-dtags -Wl,--rpath,$ROCM_PATH/lib:$ROCM_PATH/lib64"

usage() {
  echo -e "AQLProfile Build Script Usage:"
  echo -e "\nTo run ./build.sh PARAMs, PARAMs can be the following:"
  echo -e "-h   | --help               For showing this message"
  echo -e "-b   | --build              For compiling"
  echo -e "-cb  | --clean-build        For full clean build"
  exit 1
}

while [ 1 ] ; do
  if [[ "$1" = "-h" || "$1" = "--help" ]] ; then
    usage
    exit 1
  elif [[ "$1" = "-b" || "$1" = "--build" ]] ; then
    TO_CLEAN=no
    shift
  elif [[ "$1" = "-cb" || "$1" = "--clean-build" ]] ; then
    TO_CLEAN=yes
    shift
  elif [[ "$1" = "-"* || "$1" = "--"* ]] ; then
    echo -e "Wrong option \"$1\", Please use the following options:\n"
    usage
    exit 1
  else
    break
  fi
done

umask 022

if [ -z "$AQLPROFILE_ROOT" ]; then AQLPROFILE_ROOT=$SRC_DIR; fi
if [ -z "$BUILD_DIR" ] ; then BUILD_DIR=build; fi
if [ -z "$BUILD_TYPE" ] ; then BUILD_TYPE="RelWithDebInfo"; fi
if [ -z "$PACKAGE_ROOT" ] ; then PACKAGE_ROOT=$ROCM_PATH; fi
if [ -z "$PREFIX_PATH" ] ; then PREFIX_PATH=$PACKAGE_ROOT; fi
if [ -z "$HIP_VDI" ] ; then HIP_VDI=0; fi
if [ -n "$ROCM_RPATH" ] ; then LD_RUNPATH_FLAG=" -Wl,--enable-new-dtags -Wl,--rpath,${ROCM_RPATH}"; fi
if [ -z "$TO_CLEAN" ] ; then TO_CLEAN=yes; fi
if [ -z "$GPU_LIST" ] ; then GPU_LIST="gfx900 gfx906 gfx908 gfx90a gfx942 gfx950 gfx1030 gfx1100 gfx1101 gfx1102 gfx1031 gfx1150 gfx1151"; fi

AQLPROFILE_ROOT=$(cd $AQLPROFILE_ROOT && echo $PWD)

if [ "$TO_CLEAN" = "yes" ] ; then rm -rf $BUILD_DIR; fi
mkdir -p $BUILD_DIR
pushd $BUILD_DIR

cmake \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE:-'RelWithDebInfo'} \
    -DCMAKE_PREFIX_PATH="$PREFIX_PATH" \
    -DCMAKE_INSTALL_PREFIX="$PACKAGE_ROOT" \
    -DCMAKE_SHARED_LINKER_FLAGS="$LD_RUNPATH_FLAG" \
    -DCPACK_PACKAGING_INSTALL_PREFIX=$PACKAGE_ROOT \
    -DCPACK_GENERATOR=${CPACKGEN:-'DEB;RPM'} \
    -DCMAKE_INSTALL_RPATH=${ROCM_RPATH} \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE \
    -DCPACK_GENERATOR="STGZ" \
    -DGPU_TARGETS="$GPU_LIST" \
    -DAQLPROFILE_BUILD_TESTS=OFF \
    -DCPACK_OBJCOPY_EXECUTABLE="${PACKAGE_ROOT}/llvm/bin/llvm-objcopy" \
    -DCPACK_READELF_EXECUTABLE="${PACKAGE_ROOT}/llvm/bin/llvm-readelf" \
    -DCPACK_STRIP_EXECUTABLE="${PACKAGE_ROOT}/llvm/bin/llvm-strip" \
    -DCPACK_OBJDUMP_EXECUTABLE="${PACKAGE_ROOT}/llvm/bin/llvm-objdump" \
    $AQLPROFILE_ROOT

popd

MAKE_OPTS="-j -C $AQLPROFILE_ROOT/$BUILD_DIR"

cmake --build "$BUILD_DIR" -- $MAKE_OPTS all mytest
cmake --build "$BUILD_DIR" -- $MAKE_OPTS test
cmake --build "$BUILD_DIR" -- $MAKE_OPTS package

exit 0
