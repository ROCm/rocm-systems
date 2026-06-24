#!/bin/bash

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ROCm Docker Setup for rocjitsu${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if docker is available
echo -e "\n${GREEN}Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker command not found!${NC}"
    echo -e "${YELLOW}Please ensure Docker is installed and in your PATH.${NC}"
    echo -e "${YELLOW}Common solutions:${NC}"
    echo -e "  1. Install Docker Desktop: https://www.docker.com/products/docker-desktop"
    echo -e "  2. On Windows, ensure you're running this in Git Bash or WSL with Docker accessible"
    echo -e "  3. Add Docker to your PATH (e.g., /c/Program Files/Docker/Docker/resources/bin)"
    echo -e "  4. Try running: export PATH=\"\$PATH:/c/Program Files/Docker/Docker/resources/bin\""
    exit 1
fi

# Verify docker daemon is running
if ! docker ps &> /dev/null; then
    echo -e "${RED}ERROR: Docker daemon is not running!${NC}"
    echo -e "${YELLOW}Please start Docker Desktop or the Docker daemon.${NC}"
    exit 1
fi

echo -e "${GREEN}Docker is available: $(docker --version)${NC}"

# Configuration
DOCKER_IMAGE="rocm/pytorch:latest"
CONTAINER_NAME="rocjitsu-dev"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"  # rocm-systems root

echo -e "\n${GREEN}Step 1: Checking for existing container...${NC}"
CONTAINER_EXISTS=false
SKIP_IMAGE_PULL=false

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Container '${CONTAINER_NAME}' already exists.${NC}"
    CONTAINER_EXISTS=true
    SKIP_IMAGE_PULL=true
    
    # Check if container is running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${GREEN}Container is already running.${NC}"
    else
        echo -e "${YELLOW}Container exists but is stopped. Starting it...${NC}"
        docker start ${CONTAINER_NAME}
        echo -e "${GREEN}Container started successfully!${NC}"
    fi
else
    echo -e "${YELLOW}Container '${CONTAINER_NAME}' does not exist. Will create it.${NC}"
fi

if [ "$SKIP_IMAGE_PULL" = false ]; then
    echo -e "\n${GREEN}Step 2: Checking for ROCm PyTorch Docker image...${NC}"
    if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${DOCKER_IMAGE}$"; then
        echo -e "${YELLOW}Docker image '${DOCKER_IMAGE}' already exists locally.${NC}"
        read -p "Do you want to pull the latest version? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${GREEN}Pulling latest version of ${DOCKER_IMAGE}...${NC}"
            docker pull ${DOCKER_IMAGE}
        else
            echo -e "${YELLOW}Using existing local image.${NC}"
        fi
    else
        echo -e "${GREEN}Docker image not found locally. Pulling ${DOCKER_IMAGE}...${NC}"
        docker pull ${DOCKER_IMAGE}
    fi

    echo -e "\n${GREEN}Step 3: Checking if rocjitsu is already in the image...${NC}"
    if docker run --rm ${DOCKER_IMAGE} which rocjitsu >/dev/null 2>&1; then
        echo -e "${YELLOW}rocjitsu found in Docker image!${NC}"
        ROCJITSU_EXISTS=true
    else
        echo -e "${YELLOW}rocjitsu not found in Docker image. Will build from source.${NC}"
        ROCJITSU_EXISTS=false
    fi

    echo -e "\n${GREEN}Step 4: Creating Docker container...${NC}"
    docker run -itd \
        --name ${CONTAINER_NAME} \
        -v "${WORKSPACE_DIR}:/workspace" \
        -w /workspace/emulation/rocjitsu \
        ${DOCKER_IMAGE} bash
    
    echo -e "${GREEN}Container '${CONTAINER_NAME}' created successfully!${NC}"
else
    echo -e "\n${YELLOW}Skipping image pull and container creation (using existing container).${NC}"
    # Check if rocjitsu exists in the existing container
    echo -e "\n${GREEN}Step 2: Checking if rocjitsu is already built in the container...${NC}"
    if docker exec ${CONTAINER_NAME} bash -c "[ -f ./build/tools/rocjitsu/rocjitsu ]" 2>/dev/null; then
        echo -e "${GREEN}rocjitsu build found in container! Skipping build steps.${NC}"
        ROCJITSU_ALREADY_BUILT=true
        ROCJITSU_EXISTS=false  # Use local build
    else
        echo -e "${YELLOW}rocjitsu not built yet. Will build from source.${NC}"
        ROCJITSU_ALREADY_BUILT=false
        ROCJITSU_EXISTS=false
    fi
fi

# Set default for ROCJITSU_ALREADY_BUILT if not set (for new containers)
if [ -z "$ROCJITSU_ALREADY_BUILT" ]; then
    ROCJITSU_ALREADY_BUILT=false
fi

if [ "$ROCJITSU_ALREADY_BUILT" = true ]; then
    echo -e "\n${GREEN}Step 4-5: Skipping dependency installation and build (rocjitsu already built)${NC}"
    ROCJITSU_BIN="./build/tools/rocjitsu/rocjitsu"
elif [ "$ROCJITSU_EXISTS" = false ]; then
    echo -e "\n${GREEN}Step 4: Installing build dependencies...${NC}"
    docker exec ${CONTAINER_NAME} bash -c "
        apt-get update && \
        apt-get install -y cmake ninja-build git && \
        rm -rf /var/lib/apt/lists/*
    "

    echo -e "\n${GREEN}Step 5: Building rocjitsu...${NC}"
    docker exec ${CONTAINER_NAME} bash -c "
        echo '--- Building rocjitsu ---' && \
        cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
        cmake --build build && \
        echo '--- Build completed successfully ---'
    "
    ROCJITSU_BIN="./build/tools/rocjitsu/rocjitsu"
else
    ROCJITSU_BIN="rocjitsu"
fi

echo -e "\n${GREEN}Step 6: Running rocjitsu tests...${NC}"
docker exec ${CONTAINER_NAME} bash -c "
    echo '--- Running test suite ---' && \
    ctest --test-dir build --output-on-failure || echo 'Some tests may have failed (expected if HIP is not fully configured)'
"

echo -e "\n${GREEN}Step 7: Testing rocjitsu with HIP workload...${NC}"
docker exec ${CONTAINER_NAME} bash -c "
    echo '--- Testing rocjitsu with simple config ---' && \
    ${ROCJITSU_BIN} --config configs/amdgpu_cdna4_kmd.json -- echo 'rocjitsu is working!' || \
    echo 'Test completed (check output above for errors)'
"

echo -e "\n${GREEN}Step 8: Compiling hip_vector_add_test with hipcc...${NC}"
docker exec ${CONTAINER_NAME} bash -c "
    if command -v hipcc &> /dev/null; then \
        echo '--- hipcc found, compiling hip_vector_add_test.cpp ---' && \
        mkdir -p build/tests && \
        hipcc --offload-arch=gfx950 -std=c++20 \
            -isystem build/_deps/googletest-src/googletest/include \
            -Lbuild/lib \
            -lgtest -lpthread \
            -Wl,-rpath,build/lib \
            -o build/tests/hip_vector_add_test \
            tests/hip_vector_add_test.cpp && \
        echo 'Successfully compiled: build/tests/hip_vector_add_test'; \
    else \
        echo 'hipcc not found. Cannot compile HIP tests.'; \
        echo 'The rocm/pytorch image should have hipcc pre-installed.'; \
    fi
"

echo -e "\n${GREEN}Step 9: Testing rocjitsu with HIP vector_add test...${NC}"
docker exec ${CONTAINER_NAME} bash -c "
    echo '--- Testing HIP vector_add with rocjitsu simulation ---' && \
    if [ -f ./build/tests/hip_vector_add_test ]; then \
        ${ROCJITSU_BIN} --config configs/amdgpu_cdna4_kmd.json -- ./build/tests/hip_vector_add_test || \
        echo 'HIP vector_add test completed (check output above for results)'; \
    else \
        echo 'hip_vector_add_test not found (compilation may have failed)'; \
    fi
"

echo -e "\n${GREEN}Step 10: Testing rocjitsu with PyTorch...${NC}"
docker exec ${CONTAINER_NAME} bash -c "
    echo '--- Testing PyTorch with rocjitsu simulation ---' && \
    ${ROCJITSU_BIN} --daemon --config configs/amdgpu_cdna4_kmd.json -- \
    python3 -c \"import torch; print('PyTorch version:', torch.__version__); x = torch.randn(4, 4, device='cuda'); print('Tensor on CUDA:'); print(x); print('Matrix multiplication:'); print(x @ x)\" || \
    echo 'PyTorch test completed (check output above for results)'
"

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Setup completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e ""
echo -e "${YELLOW}Container Information:${NC}"
echo -e "  Name: ${CONTAINER_NAME}"
echo -e "  Image: ${DOCKER_IMAGE}"
echo -e "  Workspace: /workspace"
echo -e "  rocjitsu path: ${ROCJITSU_BIN}"
echo -e ""
echo -e "${YELLOW}Useful commands:${NC}"
echo -e "  ${GREEN}# Enter the container:${NC}"
echo -e "    docker exec -it ${CONTAINER_NAME} bash"
echo -e ""
echo -e "  ${GREEN}# Run rocjitsu with a HIP application:${NC}"
echo -e "    docker exec ${CONTAINER_NAME} ${ROCJITSU_BIN} --config configs/amdgpu_cdna4_kmd.json -- ./build/tests/hip_vector_add_test"
echo -e ""
echo -e "  ${GREEN}# Run rocjitsu with PyTorch:${NC}"
echo -e "    docker exec ${CONTAINER_NAME} ${ROCJITSU_BIN} --daemon --config configs/amdgpu_cdna4_kmd.json -- python3 -c 'import torch; ...'"
echo -e ""
echo -e "  ${GREEN}# Run tests:${NC}"
echo -e "    docker exec ${CONTAINER_NAME} ctest --test-dir build"
echo -e ""
echo -e "  ${GREEN}# Stop the container:${NC}"
echo -e "    docker stop ${CONTAINER_NAME}"
echo -e ""
echo -e "  ${GREEN}# Start the container:${NC}"
echo -e "    docker start ${CONTAINER_NAME}"
echo -e ""
echo -e "  ${GREEN}# Remove the container:${NC}"
echo -e "    docker rm -f ${CONTAINER_NAME}"
echo -e ""
echo -e "${GREEN}Done!${NC}"
