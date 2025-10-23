#!/bin/bash

# Build script for rocprofiler-sdk CI dependency Docker images
# This script builds optimized Docker images with pre-installed dependencies
# for faster CI execution

set -e

# Configuration
REGISTRY="docker.io/rocm"
BASE_TAG="rocprofiler-private"
BUILD_DATE=$(date -u +"%Y%m%d")
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to build and tag stage-1 (deps) image
build_stage1_image() {
    local dockerfile="$1"
    local os_name="$2"
    local os_version="$3"

    echo "Building stage-1 (deps) ${os_name}-${os_version} image..."

    docker build \
        -f "${SCRIPT_DIR}/${dockerfile}" \
        -t "${REGISTRY}/${BASE_TAG}:${os_name}-${os_version}-${BUILD_DATE}" \
        -t "${REGISTRY}/${BASE_TAG}:${os_name}-${os_version}-latest-testing" \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg GIT_HASH="${GIT_HASH}" \
        --build-arg ROCM_SYSTEMS_REF="${ROCM_SYSTEMS_REF:-develop}" \
        --build-arg ROCM_SYSTEMS_REV="${ROCM_SYSTEMS_REV:-}" \
        "${SCRIPT_DIR}"

    echo "Successfully built stage-1 ${os_name}-${os_version}"
}

# Get latest therock tarball S3 key for a given GPU type (e.g., gfx94X)
get_latest_tarball_key() {
    local gpu_type="$1"
    # Use native AWS CLI; bucket is public
    aws s3api list-objects-v2 \
        --bucket therock-nightly-tarball \
        --no-sign-request \
        --output json \
        --query "sort_by(Contents[?contains(Key, 'linux-${gpu_type}')], &LastModified)[-1].Key" | \
        tr -d '"\r\n'
}

# Function to build stage-2 (final) image for a specific OS and GPU
build_stage2_image() {
    local os_tag="$1"
    local gpu="$2"

    echo "Building stage-2 (final) ${os_tag}-${gpu} using ${TARBALL_KEYS["${gpu}"]}..."

    docker build \
        -f "${SCRIPT_DIR}/Dockerfile.stages" \
        --build-arg BASE_IMAGE="${REGISTRY}/${BASE_TAG}:${os_tag}-latest-testing" \
        --build-arg GPU_TYPE="${gpu}" \
        --build-arg TARBALL_KEY="${TARBALL_KEYS["${gpu}"]}" \
        --build-arg ROCM_SYSTEMS_REF="${ROCM_SYSTEMS_REF:-develop}" \
        --build-arg ROCM_SYSTEMS_REV="${ROCM_SYSTEMS_REV:-}" \
        --build-arg BUILD_ROCPROFILER_REGISTER="${BUILD_ROCPROFILER_REGISTER:-true}" \
        --build-arg BUILD_ROCR_RUNTIME="${BUILD_ROCR_RUNTIME:-true}" \
        --build-arg BUILD_AQLPROFILE="${BUILD_AQLPROFILE:-true}" \
        --build-arg BUILD_CLR="${BUILD_CLR:-true}" \
        --build-arg ROCDECODE_REF="${ROCDECODE_REF:-release/rocm-rel-7.0}" \
        --build-arg ROCDECODE_REV="${ROCDECODE_REV:-}" \
        --build-arg ROCJPEG_REF="${ROCJPEG_REF:-release/rocm-rel-7.0}" \
        --build-arg ROCJPEG_REV="${ROCJPEG_REV:-}" \
        --build-arg BUILD_ROCDECODE="${BUILD_ROCDECODE:-true}" \
        --build-arg BUILD_ROCJPEG="${BUILD_ROCJPEG:-true}" \
        -t "${REGISTRY}/${BASE_TAG}:${os_tag}-${gpu}-${BUILD_DATE}" \
        -t "${REGISTRY}/${BASE_TAG}:${os_tag}-${gpu}-latest-testing" \
        "${SCRIPT_DIR}"
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
    echo "  -g, --gpus      Comma-separated GPU list (gfx94X,gfx950,gfx110X,gfx120X). Default: all"
    echo "      --skip-rocm Skip ROCm stages (2-4) and only build Stage 1"
    echo "      --skip-missing-tarballs Skip GPU types that don't have available tarballs"
    echo ""
    echo "Distributions:"
    echo "  ubuntu-22.04    Build Ubuntu 22.04 image"
    echo "  ubuntu-24.04    Build Ubuntu 24.04 image"
    echo "  almalinux-8.10 Build AlmaLinux 8.10 image"
    echo "  almalinux-10         Build RHEL 10.0 image"
    echo "  sles-15.7       Build SLES 15.7 image"
    echo ""
    echo "Examples:"
    echo "  $0 --all                      # Build all distributions"
    echo "  $0 ubuntu-22.04 almalinux-10      # Build only Ubuntu 22.04 and RHEL 10"
    echo "  $0 --push ubuntu-22.04       # Build and push Ubuntu 22.04 image"
}

# Parse command line arguments
PUSH_IMAGES=false
BUILD_ALL=true
DISTRIBUTIONS=()
GPU_TYPES=("gfx94X" "gfx950" "gfx110X" "gfx120X")
SKIP_ROCM=false
SKIP_MISSING_TARBALLS=false

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
        -g|--gpus)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --gpus requires a value"; exit 1
            fi
            IFS=',' read -r -a GPU_TYPES <<< "$2"
            shift 2
            ;;
        --skip-rocm)
            SKIP_ROCM=true
            shift
            ;;
        --skip-missing-tarballs)
            SKIP_MISSING_TARBALLS=true
            shift
            ;;
        ubuntu-22.04|ubuntu-24.04|almalinux-8.10|almalinux-10|sles-15.7)
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
    DISTRIBUTIONS=("ubuntu-22.04" "ubuntu-24.04" "almalinux-8.10" "almalinux-10" "sles-15.7")
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
echo "GPUs: ${GPU_TYPES[*]}"
echo ""

# Resolve latest therock tarball keys once per GPU
declare -A TARBALL_KEYS
AVAILABLE_GPUS=()
for gpu in "${GPU_TYPES[@]}"; do
    echo "Resolving latest tarball for ${gpu}..."
    key=$(get_latest_tarball_key "${gpu}")
    if [[ -z "${key}" || "${key}" == "null" ]]; then
        echo "Warning: Could not resolve tarball for ${gpu}"
        if [[ ${SKIP_MISSING_TARBALLS} == false ]]; then
            echo "Available tarballs in bucket:"
            aws s3api list-objects-v2 \
                --bucket therock-nightly-tarball \
                --no-sign-request \
                --output json \
                --query "Contents[].Key" | head -20
            exit 1
        else
            echo "Skipping ${gpu} due to missing tarball"
            continue
        fi
    fi
    TARBALL_KEYS["${gpu}"]="${key}"
    AVAILABLE_GPUS+=("${gpu}")
    echo "${gpu} -> ${key}"
done

# Update GPU_TYPES to only include available GPUs
if [[ ${SKIP_MISSING_TARBALLS} == true ]]; then
    GPU_TYPES=("${AVAILABLE_GPUS[@]}")
    echo "Updated GPU list to available tarballs: ${GPU_TYPES[*]}"
fi

for dist in "${DISTRIBUTIONS[@]}"; do
    case $dist in
        ubuntu-22.04)
            build_stage1_image "Dockerfile.ubuntu-22.04" "ubuntu" "22.04"
            if [[ ${SKIP_ROCM} == false ]]; then
                for gpu in "${GPU_TYPES[@]}"; do
                    build_stage2_image "ubuntu-22.04" "${gpu}"
                done
            fi
            ;;
        ubuntu-24.04)
            build_stage1_image "Dockerfile.ubuntu-24.04" "ubuntu" "24.04"
            if [[ ${SKIP_ROCM} == false ]]; then
                for gpu in "${GPU_TYPES[@]}"; do
                    build_stage2_image "ubuntu-24.04" "${gpu}"
                done
            fi
            ;;
        almalinux-8.10)
            build_stage1_image "Dockerfile.almalinux-8.10" "almalinux" "8.10"
            if [[ ${SKIP_ROCM} == false ]]; then
                for gpu in "${GPU_TYPES[@]}"; do
                    build_stage2_image "almalinux-8.10" "${gpu}"
                done
            fi
            ;;
        almalinux-10)
            build_stage1_image "Dockerfile.almalinux-10" "almalinux" "10"
            if [[ ${SKIP_ROCM} == false ]]; then
                for gpu in "${GPU_TYPES[@]}"; do
                    build_stage2_image "almalinux-10" "${gpu}"
                done
            fi
            ;;
        sles-15.7)
            build_stage1_image "Dockerfile.sles-15.7" "sles" "15.7"
            if [[ ${SKIP_ROCM} == false ]]; then
                for gpu in "${GPU_TYPES[@]}"; do
                    build_stage2_image "sles-15.7" "${gpu}"
                done
            fi
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
            ubuntu-22.04)
                docker push "${REGISTRY}/${BASE_TAG}:ubuntu-22.04-${BUILD_DATE}"
                docker push "${REGISTRY}/${BASE_TAG}:ubuntu-22.04-latest-testing"
                for gpu in "${GPU_TYPES[@]}"; do
                    docker push "${REGISTRY}/${BASE_TAG}:ubuntu-22.04-${gpu}-${BUILD_DATE}"
                    docker push "${REGISTRY}/${BASE_TAG}:ubuntu-22.04-${gpu}-latest-testing"
                done
                ;;
            ubuntu-24.04)
                docker push "${REGISTRY}/${BASE_TAG}:ubuntu-24.04-${BUILD_DATE}"
                docker push "${REGISTRY}/${BASE_TAG}:ubuntu-24.04-latest-testing"
                for gpu in "${GPU_TYPES[@]}"; do
                    docker push "${REGISTRY}/${BASE_TAG}:ubuntu-24.04-${gpu}-${BUILD_DATE}"
                    docker push "${REGISTRY}/${BASE_TAG}:ubuntu-24.04-${gpu}-latest-testing"
                done
                ;;
            almalinux-8.10)
                docker push "${REGISTRY}/${BASE_TAG}:almalinux-8.10-${BUILD_DATE}"
                docker push "${REGISTRY}/${BASE_TAG}:almalinux-8.10-latest-testing"
                for gpu in "${GPU_TYPES[@]}"; do
                    docker push "${REGISTRY}/${BASE_TAG}:almalinux-8.10-${gpu}-${BUILD_DATE}"
                    docker push "${REGISTRY}/${BASE_TAG}:almalinux-8.10-${gpu}-latest-testing"
                done
                ;;
            almalinux-10)
                docker push "${REGISTRY}/${BASE_TAG}:almalinux-10-${BUILD_DATE}"
                docker push "${REGISTRY}/${BASE_TAG}:almalinux-10-latest-testing"
                for gpu in "${GPU_TYPES[@]}"; do
                    docker push "${REGISTRY}/${BASE_TAG}:almalinux-10-${gpu}-${BUILD_DATE}"
                    docker push "${REGISTRY}/${BASE_TAG}:almalinux-10-${gpu}-latest-testing"
                done
                ;;
            sles-15.7)
                docker push "${REGISTRY}/${BASE_TAG}:sles-15.7-${BUILD_DATE}"
                docker push "${REGISTRY}/${BASE_TAG}:sles-15.7-latest-testing"
                for gpu in "${GPU_TYPES[@]}"; do
                    docker push "${REGISTRY}/${BASE_TAG}:sles-15.7-${gpu}-${BUILD_DATE}"
                    docker push "${REGISTRY}/${BASE_TAG}:sles-15.7-${gpu}-latest-testing"
                done
                ;;
        esac
    done
fi

echo ""
echo "Build completed successfully!"
echo ""
echo "Available images:"
docker images | grep "${REGISTRY}/${BASE_TAG}" | head -20

echo ""
echo "To use these images in CI, update your workflow files to use (examples):"
echo "  ${REGISTRY}/${BASE_TAG}:ubuntu-22.04-gfx94X-latest-testing"
echo "  ${REGISTRY}/${BASE_TAG}:ubuntu-24.04-gfx94X-latest-testing"
echo "  ${REGISTRY}/${BASE_TAG}:almalinux-8.10-gfx94X-latest-testing"
echo "  ${REGISTRY}/${BASE_TAG}:almalinux-10-gfx110X-latest-testing"
echo "  ${REGISTRY}/${BASE_TAG}:sles-15.7-gfx120X-latest-testing"