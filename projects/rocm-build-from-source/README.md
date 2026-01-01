# Building ROCm Components from Source

This guide walks through building `rocr-runtime`, `clr`, `hip`, `hip-tests`, `rocprofiler-sdk`, and `rocprofiler-systems` from source inside a Docker container.

## Prerequisites

- Docker installed with GPU support
- AMD GPU hardware (MI300X, MI250, etc.)
- Host machine with ROCm-compatible drivers

---

## Quick Start (Automated Script)

```bash
# On host machine
mkdir -p ~/rocm-workspace
docker pull rocm/tensorflow:latest

docker run -it \
    --name rocm-tf-dev \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --shm-size 16G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --cap-add=IPC_LOCK \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 \
    --ulimit nofile=1048576:1048576 \
    -v /tmp/rocprof_docker:/tmp/rocprof \
    -v $HOME/rocm-workspace:/workspace \
    -v $HOME/docker-home:/root \
    -e ROCPROF_TMPDIR=/tmp/rocprof \
    -e ROCPROF_OUTPUT_PATH=/tmp/rocprof \
    -w /workspace \
    rocm/tensorflow:latest bash

# Inside container - download and run the script
wget https://raw.githubusercontent.com/ROCm/rocm-systems/develop/projects/rocm-build-from-source/setup_rocm_from_source.sh
chmod +x setup_rocm_from_source.sh
./setup_rocm_from_source.sh --all
```

---

## Manual Step-by-Step Instructions

### Step 1: Pull and Run Docker Container

```bash
# Pull latest TensorFlow ROCm image
docker pull rocm/tensorflow:latest

# Run container with required flags
docker run -it \
    --name rocm-tf-dev \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --shm-size 16G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --cap-add=IPC_LOCK \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 \
    --ulimit nofile=1048576:1048576 \
    -v /tmp/rocprof_docker:/tmp/rocprof \
    -v $HOME/rocm-workspace:/workspace \
    -v $HOME/docker-home:/root \
    -e ROCPROF_TMPDIR=/tmp/rocprof \
    -e ROCPROF_OUTPUT_PATH=/tmp/rocprof \
    -w /workspace \
    rocm/tensorflow:latest bash
```

#### Docker Flags Explained

| Flag | Purpose |
|------|---------|
| `--name rocm-tf-dev` | Name container for easy restart |
| `--ulimit memlock=-1:-1` | **Critical:** Allows mmap for profiling |
| `--cap-add=IPC_LOCK` | Allows memory locking |
| `-v` mounts | Persist data between sessions |

### Step 2: Install CMake First (Required)

```bash
apt-get update
apt-get install -y make build-essential python3-pip

# Install latest CMake via pip (required >= 3.21)
pip install --break-system-packages cmake --upgrade

# Add pip binaries to PATH
export PATH="${HOME}/.local/bin:${PATH}"

# Verify installation
cmake --version  # Should show 3.21 or higher
```

### Step 3: Clone rocm-systems Repository

```bash
cd /workspace
git clone https://github.com/ROCm/rocm-systems.git
cd rocm-systems
```

### Step 4: Install Common Dependencies

```bash
apt-get update

# Build essentials for rocr-runtime
apt-get install -y \
    g++ \
    libelf-dev \
    libdrm-amdgpu-dev \
    pkg-config \
    python3 python3-pip \
    xxd  # Required for rocr-runtime shader generation

# IMPORTANT: ROCm LLVM is required to avoid version mismatch with system Clang
apt-get install -y \
    rocm-core \
    rocm-llvm-dev

pip3 install CppHeaderParser

# Set CMAKE_PREFIX_PATH for all subsequent builds
export CMAKE_PREFIX_PATH=/opt/rocm
```

> **Note:** `rocm-llvm-dev` provides LLVM/Clang 20 for ROCm. Without this, CMake will find the system's Clang 18, causing a version mismatch error.

### Step 5: Build ROCR-Runtime

```bash
cd /workspace/rocm-systems/projects/rocr-runtime
mkdir build && cd build

cmake \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_BUILD_TYPE=Release \
    ..

make -j$(nproc)
make install
```

### Step 6: Build CLR + HIP

```bash
# Install SIMDe (SIMD Everywhere) - required for rocclr
apt-get install -y libsimde-dev || {
    # If not available via apt, install from source
    cd /tmp
    git clone https://github.com/simd-everywhere/simde.git
    cd simde && mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
    make install
}

export HIP_DIR="/workspace/rocm-systems/projects/hip"

cd /workspace/rocm-systems/projects/clr
mkdir build && cd build

cmake .. \
    -DCLR_BUILD_HIP=ON \
    -DCLR_BUILD_OCL=OFF \
    -DHIP_COMMON_DIR=$HIP_DIR \
    -DHIP_PLATFORM=amd \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm

make -j$(nproc)
make install
```

### Step 7: Build ROCprofiler-SDK

```bash
# Install SDK dependencies
apt-get install -y libdw-dev libsqlite3-dev

# Set compiler (confirm GCC version first)
g++ --version  # Check available version
export CXX=/usr/bin/g++-13
export CC=/usr/bin/gcc-13

# Install Python requirements
export PATH=${HOME}/.local/bin:${PATH}
python3 -m pip install -r /workspace/rocm-systems/projects/rocprofiler-sdk/requirements.txt

# Build
cd /workspace/rocm-systems
cmake \
    -B rocprofiler-sdk-build \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    projects/rocprofiler-sdk

cmake --build rocprofiler-sdk-build --target all --parallel $(nproc)
cmake --build rocprofiler-sdk-build --target install

# Add ROCm bin to PATH (required for rocprofv3)
export PATH=$PATH:/opt/rocm/bin

# Make permanent (optional)
echo 'export PATH=$PATH:/opt/rocm/bin' >> ~/.bashrc
```

### Step 8: Build ROCprofiler-Systems

```bash
# Install dependencies
apt-get install -y \
    autoconf autotools-dev bash-completion bison build-essential \
    bzip2 chrpath curl environment-modules flex gettext git-core gnupg2 \
    gzip iproute2 libiberty-dev libpapi-dev libpfm4-dev libsqlite3-dev libtool \
    locales lsb-release m4 ninja-build python3-pip software-properties-common \
    texinfo unzip wget vim zip zlib1g-dev

# Install perfetto (cmake should already be latest from Step 2)
pip install --break-system-packages perfetto

# Build (with debug enabled using preset)
cd /workspace/rocm-systems/projects/rocprofiler-systems

# Use debug preset (CMAKE_BUILD_TYPE=Debug, ROCPROFSYS_BUILD_DEBUG=ON)
cmake --preset debug
cmake --build build/debug --parallel 8
cmake --build build/debug --target install

# Setup environment
source /opt/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh
```

---

## Testing the Build

### Create Example Script

```bash
cat > /workspace/example.py << 'EOF'
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
predictions = model(x_train[:1]).numpy()
tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)
EOF
```

### Test with rocprofv3

```bash
cd /workspace

# Ensure ROCm bin is in PATH
export PATH=$PATH:/opt/rocm/bin

# Create output directories first
mkdir -p /tmp/rocprof
export ROCPROF_TMPDIR=/tmp/rocprof
export ROCPROF_OUTPUT_PATH=/tmp/rocprof

# Run profiler
rocprofv3 -s -o /tmp/rocprof/output -- python3.12 example.py
```

**Expected output:**
- Training completes with ~97% accuracy
- SQLite database generated at `/tmp/rocprof/output_results.db`
- Final line: `[rocprofv3] tool finalization :: XX.XX sec`

**If you get a segfault:** Try using the pre-installed version:
```bash
/opt/rocm/bin/rocprofv3 -s -o /tmp/rocprof/output -- python3.12 example.py
```

### Test with rocprof-sys-run

```bash
source /opt/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh
rocprof-sys-run -- python3.12 example.py
```

**Note:** This may crash or require additional configuration.

---

## Container Management

```bash
# Exit container
exit

# Restart container later (preserves all changes)
docker start -ai rocm-tf-dev

# Execute command in running container
docker exec -it rocm-tf-dev bash

# View container logs
docker logs rocm-tf-dev

# Remove container (to start fresh)
docker rm rocm-tf-dev
```

---

## Troubleshooting

### Issue: `mmap failed with errno 22`

**Solution:** Ensure Docker is run with `--ulimit memlock=-1:-1`

### Issue: LLVM version mismatch

**Solution:** Set `CMAKE_PREFIX_PATH=/opt/rocm` when running cmake

### Issue: `xxd not found`

**Solution:** `apt-get install xxd`

### Issue: `Segmentation fault` when running rocprofv3

**Cause:** Version mismatch between built ROCm components and the pre-installed ROCm in the container.

**Solutions:**
1. **Use the pre-installed rocprofv3** (recommended for testing):
   ```bash
   # Use the container's pre-installed version instead of built version
   /opt/rocm/bin/rocprofv3 -s -o /tmp/rocprof/output -- python3.12 example.py
   ```

2. **Ensure output directories exist:**
   ```bash
   mkdir -p /tmp/rocprof
   export ROCPROF_TMPDIR=/tmp/rocprof
   export ROCPROF_OUTPUT_PATH=/tmp/rocprof
   ```

3. **Rebuild with clean install prefix** (if you need the built version):
   ```bash
   # Consider using a different install prefix to avoid conflicts
   # CMAKE_INSTALL_PREFIX=/opt/rocm-custom
   ```

### Issue: rocprof-sys-run crashes

**Solution:** This is a known issue with some configurations. Use `rocprofv3` instead.

### Issue: Output file not found

**Cause:** rocprofv3 crashed before generating output files.

**Solution:** 
1. Create output directories before running:
   ```bash
   mkdir -p /tmp/rocprof
   ```
2. Check if rocprofv3 works with a simple test:
   ```bash
   rocprofv3 --version
   ```
3. Use the pre-installed version if the built version crashes:
   ```bash
   /opt/rocm/bin/rocprofv3 -s -o /tmp/output -- your_command
   ```

---

## References

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Build Instructions](https://rocm.docs.amd.com/projects/HIP/en/latest/install/build.html)
- [ROCprofiler-SDK Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)
- [ROCprofiler-Systems Documentation](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/)

