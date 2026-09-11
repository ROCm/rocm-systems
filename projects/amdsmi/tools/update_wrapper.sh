#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# this program generates py-interface/amdsmi_wrapper.py

set -eu

# get current dir
DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd -P )

# override by calling this script with:
# DOCKER_NAME=yourdockername ./update_wrapper.sh
DOCKER_NAME="${DOCKER_NAME:-dmitriigalantsev/amdsmi_wrapper_updater}"

command -v docker &>/dev/null || {
    echo "Please install docker!" >&2
    exit 1
}

DOCKER_BUILDKIT=$(docker buildx version >/dev/null 2>&1 && echo 1 || echo 0)
export DOCKER_BUILDKIT

IMAGE_REF=""

build_docker_image () {
    DOCKER_DIR=$(cd "$DIR/py-interface" && pwd -P)
    DOCKERFILE="$DOCKER_DIR/Dockerfile"

    # Tag the image by Dockerfile hash; rebuild when no image matches the current one.
    DOCKERFILE_HASH=$(sha256sum "$DOCKERFILE" | cut -c1-12)
    IMAGE_REF="$DOCKER_NAME:$DOCKERFILE_HASH"

    if docker image inspect "$IMAGE_REF" >/dev/null 2>&1; then
        echo "Reusing up-to-date image: $IMAGE_REF"
    else
        echo "Dockerfile changed or image missing -- (re)building $IMAGE_REF"
        docker build "$DOCKER_DIR" -f "$DOCKERFILE" -t "$IMAGE_REF" -t "$DOCKER_NAME":latest
    fi
}

build_docker_image

ENABLE_ESMI_LIB=""
# source ENABLE_ESMI_LIB variable from the previous build if it exists
if [ -e "${DIR}/build/CMakeCache.txt" ]; then
    GREP_RESULT=$(grep "ENABLE_ESMI_LIB.*=" "${DIR}/build/CMakeCache.txt" | tail -n 1 | cut -d = -f 2)
    ENABLE_ESMI_LIB="-DENABLE_ESMI_LIB=$GREP_RESULT"
    echo "ENABLE_ESMI_LIB: [$ENABLE_ESMI_LIB]"
fi

DOCKER_TTY_FLAGS=(-i)
if [ -t 0 ]; then DOCKER_TTY_FLAGS=(-t -i); fi

docker run --rm "${DOCKER_TTY_FLAGS[@]}" --volume "$DIR":/src:rw "$IMAGE_REF" bash -c "
cp -r /src /tmp/src \
&& cd /tmp/src \
&& rm -rf build .cache \
&& cmake -B build -DBUILD_WRAPPER=ON $ENABLE_ESMI_LIB \
&& make -C build -j \$(nproc) \
&& cp /tmp/src/py-interface/amdsmi_wrapper.py /src/py-interface/amdsmi_wrapper.py \
&& chown --reference /src/py-interface/CMakeLists.txt /src/py-interface/amdsmi_wrapper.py"

echo -e "Generated new wrapper!
[$DIR/py-interface/amdsmi_wrapper.py]"
