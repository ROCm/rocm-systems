#!/usr/bin/env bash
# Builds the reproducers in this directory against a hipFile build tree.
#
#   usage: build.sh HIPFILE_BUILD_DIR [OUTPUT_DIR]
#
# HIPFILE_BUILD_DIR is the directory hipFile was built in (it holds
# src/amd_detail/libhipfile.so). The programs go in OUTPUT_DIR, which defaults
# to ./bin, and find that libhipfile.so through an rpath, ahead of any
# installed one. Set ROCM_PATH if ROCm is not in /opt/rocm, and EXTRA_FLAGS to
# add compiler flags, such as --offload-arch=gfx1030 on a machine without the
# GPU.

here=$(cd "$(dirname "$0")" && pwd)
build=${1:?usage: build.sh HIPFILE_BUILD_DIR [OUTPUT_DIR]}
out=${2:-$here/bin}
rocm=${ROCM_PATH:-/opt/rocm}
lib=$(cd "$build/src/amd_detail" && pwd) || exit 1
[ -e "$lib/libhipfile.so" ] || { echo "error: no libhipfile.so in $lib" >&2; exit 1; }
mkdir -p "$out" || exit 1

# EXTRA_FLAGS is a list of flags, so it is split on purpose
read -r -a extra_flags <<<"${EXTRA_FLAGS:-}"
flags=(-std=c++20 -O2 -g -Wall -Wextra -I"$here" -I"$here/../include" "${extra_flags[@]}")
# An old-style rpath, which is searched before LD_LIBRARY_PATH
link=(-L"$lib" -lhipfile "-Wl,--disable-new-dtags" "-Wl,-rpath,$lib" -pthread)

# Program name, source file, and the -D flags that select a variant
programs=(
    "01-write-after-hipmemset 01-write-after-hipmemset.cpp"
    "01-write-after-hipmemsetd8 01-write-after-hipmemset.cpp -DCALL=8"
    "01-write-after-hipmemsetd16 01-write-after-hipmemset.cpp -DCALL=16"
    "01-write-after-hipmemsetd32 01-write-after-hipmemset.cpp -DCALL=32"
    "02-write-after-device-to-device-hipmemcpy 02-write-after-device-to-device-hipmemcpy.cpp"
    "02-write-after-hipmemcpydtod 02-write-after-device-to-device-hipmemcpy.cpp -DCALL=2"
    "02-write-after-hipmemcpywithstream 02-write-after-device-to-device-hipmemcpy.cpp -DCALL=1"
    "02-write-after-hipmemcpywithstream-fallback 02-write-after-device-to-device-hipmemcpy.cpp -DCALL=1 -DBACKEND=1"
    "03-write-after-kernel-without-sync 03-write-after-kernel-without-sync.cpp"
    "04-write-small-after-large-hipmemcpy 04-write-small-after-large-hipmemcpy.cpp"
    "05-write-small-after-fallback-read 05-write-small-after-fallback-read.cpp"
    "06-read-then-kernel-sees-old-data 06-read-then-kernel-sees-old-data.cpp"
    "07-read-then-cpu-sees-old-data 07-read-then-cpu-sees-old-data.cpp"
)
for program in "${programs[@]}"; do
    read -r -a words <<<"$program"
    name=${words[0]}
    source=${words[1]}
    echo "building $name"
    "$rocm/bin/hipcc" "${flags[@]}" "${words[@]:2}" "$here/$source" -o "$out/$name" "${link[@]}" || exit 1
done
echo "built ${#programs[@]} programs in $out"
