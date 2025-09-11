#!/bin/bash

# Build script for rocprofiler-sdk CI dependency Docker images
# This script builds optimized Docker images with pre-installed dependencies
# for faster CI execution

set -e

# Configuration
REGISTRY="docker.io/rocm"
BASE_TAG="rocprofiler-deps"
BUILD_DATE=$(date -u +"%Y%m%d")
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Function to build and tag image
build_image() {
    local dockerfile="$1"
    local os_name="$2"
    local os_version="$3"
    
    echo "Building ${os_name}-${os_version} image..."
    
    # Build the image
    docker build \
        -f "${dockerfile}" \
        -t "${REGISTRY}/${BASE_TAG}:${os_name}-${os_version}-${BUILD_DATE}" \
        -t "${REGISTRY}/${BASE_TAG}:${os_name}-${os_version}-latest" \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg GIT_HASH="${GIT_HASH}" \
        .
    
    echo "Successfully built ${os_name}-${os_version} image"
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS] [DISTRIBUTIONS]"
    echo ""
    echo "Build rocprofiler-sdk CI dependency Docker images"
    echo ""
    echo "Options:"
    echo "  -h, --help      Show this help message"
    echo "  -p, --push      Push images to registry after building"
    echo "  -a, --all       Build all distributions (default)"
    echo ""
    echo "Distributions:"
    echo "  ubuntu          Build Ubuntu 22.04 image"
    echo "  rhel8           Build RHEL 8.8 image"
    echo "  rhel9           Build RHEL 9.5 image"
    echo "  sles            Build SLES 15.6 image"
    echo ""
    echo "Examples:"
    echo "  $0 --all              # Build all distributions"
    echo "  $0 ubuntu rhel9       # Build only Ubuntu and RHEL 9"
    echo "  $0 --push ubuntu      # Build and push Ubuntu image"
}

# Parse command line arguments
PUSH_IMAGES=false
BUILD_ALL=true
DISTRIBUTIONS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -p|--push)
            PUSH_IMAGES=true
            shift
            ;;
        -a|--all)
            BUILD_ALL=true
            shift
            ;;
        ubuntu|rhel8|rhel9|sles)
            BUILD_ALL=false
            DISTRIBUTIONS+=("$1")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Set default distributions if none specified
if [[ ${BUILD_ALL} == true ]]; then
    DISTRIBUTIONS=("ubuntu" "rhel8" "rhel9" "sles")
fi

# Verify Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running or not accessible"
    exit 1
fi

# Build images
echo "Building rocprofiler-sdk CI dependency images..."
echo "Build date: ${BUILD_DATE}"
echo "Git hash: ${GIT_HASH}"
echo "Distributions: ${DISTRIBUTIONS[*]}"
echo ""

for dist in "${DISTRIBUTIONS[@]}"; do
    case $dist in
        ubuntu)
            build_image "Dockerfile.ubuntu-22.04" "ubuntu" "22.04"
            ;;
        rhel8)
            build_image "Dockerfile.rhel-8.8" "rhel" "8.8"
            ;;
        rhel9)
            build_image "Dockerfile.rhel-9.5" "rhel" "9.5"
            ;;
        sles)
            build_image "Dockerfile.sles-15.6" "sles" "15.6"
            ;;
        *)
            echo "Warning: Unknown distribution '$dist', skipping..."
            ;;
    esac
done

# Push images if requested
if [[ ${PUSH_IMAGES} == true ]]; then
    echo ""
    echo "Pushing images to registry..."
    
    for dist in "${DISTRIBUTIONS[@]}"; do
        case $dist in
            ubuntu)
                docker push "${REGISTRY}/${BASE_TAG}:ubuntu-22.04-${BUILD_DATE}"
                docker push "${REGISTRY}/${BASE_TAG}:ubuntu-22.04-latest"
                ;;
            rhel8)
                docker push "${REGISTRY}/${BASE_TAG}:rhel-8.8-${BUILD_DATE}"
                docker push "${REGISTRY}/${BASE_TAG}:rhel-8.8-latest"
                ;;
            rhel9)
                docker push "${REGISTRY}/${BASE_TAG}:rhel-9.5-${BUILD_DATE}"
                docker push "${REGISTRY}/${BASE_TAG}:rhel-9.5-latest"
                ;;
            sles)
                docker push "${REGISTRY}/${BASE_TAG}:sles-15.6-${BUILD_DATE}"
                docker push "${REGISTRY}/${BASE_TAG}:sles-15.6-latest"
                ;;
        esac
    done
fi

echo ""
echo "Build completed successfully!"
echo ""
echo "Available images:"
docker images | grep "${REGISTRY}/${BASE_TAG}" | head -10

echo ""
echo "To use these images in CI, update your workflow files to use:"
echo "  ${REGISTRY}/${BASE_TAG}:ubuntu-22.04-latest"
echo "  ${REGISTRY}/${BASE_TAG}:rhel-8.8-latest"
echo "  ${REGISTRY}/${BASE_TAG}:rhel-9.5-latest"
echo "  ${REGISTRY}/${BASE_TAG}:sles-15.6-latest"