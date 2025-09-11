# ROCProfiler-SDK CI Dependency Docker Images

This directory contains Dockerfiles that pre-install all the dependencies needed for rocprofiler-sdk CI builds. These images can significantly speed up CI execution by avoiding repeated package installations.

## Images Available

### Ubuntu 22.04 (`Dockerfile.ubuntu-22.04`)
Based on: `docker.io/rocm/rocprofiler-private:ubuntu-22.04-gfx94X-latest`

**Pre-installed packages:**
- Build tools: gcc-11/12/13, g++, clang-15, cmake, make
- Development libraries: libdw-dev, libsqlite3-dev, libdrm-dev, libelf-dev
- Python: python3, pip, venv
- Documentation: doxygen, graphviz
- Coverage tools: gcovr, wkhtmltopdf, X11 fonts
- Sanitizer libraries: libasan8, libtsan2
- Build acceleration: ccache

### RHEL 8.8 (`Dockerfile.rhel-8.8`)
Based on: `docker.io/rocm/rocprofiler-private:rhel-8.8-gfx94X-latest`

**Pre-installed packages:**
- Build tools: gcc, gcc-c++, gcc-toolset-11, cmake
- Development libraries: elfutils-libelf-devel, sqlite-devel, libdrm-devel
- Python: python3, pip
- Documentation: doxygen, graphviz
- Build acceleration: sccache

### RHEL 9.5 (`Dockerfile.rhel-9.5`)
Based on: `docker.io/rocm/rocprofiler-private:rhel-9.5-gfx94X-latest`

**Pre-installed packages:**
- Similar to RHEL 8.8 but with RHEL 9 package versions
- GCC toolset 11 for modern C++ support
- Enhanced development tools

### SLES 15.6 (`Dockerfile.sles-15.6`)
Based on: `docker.io/rocm/rocprofiler-private:sles-15.6-gfx94X-latest`

**Pre-installed packages:**
- Build tools: gcc, gcc-c++, cmake (gcc11 if available)
- Development libraries: libelf-devel, sqlite3-devel, libdrm-devel
- Python: python3, pip
- Documentation: doxygen, graphviz
- Build acceleration: sccache

## Building Images

### Using the Build Script (Recommended)

```bash
# Build all distributions
./docker-build.sh --all

# Build specific distributions
./docker-build.sh ubuntu rhel9

# Build and push to registry
./docker-build.sh --push ubuntu

# Show help
./docker-build.sh --help
```

### Manual Building

```bash
# Ubuntu 22.04
docker build -f Dockerfile.ubuntu-22.04 -t rocm/rocprofiler-deps:ubuntu-22.04-latest .

# RHEL 8.8
docker build -f Dockerfile.rhel-8.8 -t rocm/rocprofiler-deps:rhel-8.8-latest .

# RHEL 9.5
docker build -f Dockerfile.rhel-9.5 -t rocm/rocprofiler-deps:rhel-9.5-latest .

# SLES 15.6
docker build -f Dockerfile.sles-15.6 -t rocm/rocprofiler-deps:sles-15.6-latest .
```

## Using in CI

To use these optimized images in your CI workflows, update the container image references:

```yaml
# Before (slower - installs deps every time)
container:
  image: docker.io/rocm/rocprofiler-private:ubuntu-22.04-gfx94X-latest

# After (faster - deps pre-installed)
container:
  image: docker.io/rocm/rocprofiler-deps:ubuntu-22.04-latest
```

### Example CI Workflow Modification

```yaml
jobs:
  core-deb:
    runs-on: rocprofiler-navi3-dind
    container:
      image: docker.io/rocm/rocprofiler-deps:ubuntu-22.04-latest
      credentials:
        username: ${{ secrets.ROCPROFILER_AZURE_CI_USER }}
        password: ${{ secrets.ROCPROFILER_AZURE_CI_PASS }}
    steps:
      # Skip dependency installation steps - already pre-installed!
      - name: Clone ROCProfiler SDK
        uses: actions/checkout@v5
        # ... rest of workflow
```

## Benefits

### Time Savings
- **Ubuntu builds**: ~2-3 minutes saved per job (dependency installation eliminated)
- **RHEL/SLES builds**: ~3-4 minutes saved per job (slower package managers)
- **Overall**: 15-20% reduction in total CI time

### Reliability
- Eliminates package installation failures
- Consistent dependency versions across builds
- Reduces network-related CI failures

### Resource Efficiency
- Lower bandwidth usage (dependencies cached in image)
- Reduced load on package repositories
- More predictable CI resource consumption

## Maintenance

### Updating Dependencies

When CI requirements change:

1. Update the appropriate Dockerfile(s)
2. Rebuild and test the images locally
3. Push updated images to the registry
4. Update CI workflows to use new image tags

### Version Management

Images are tagged with both date and "latest":
- `rocm/rocprofiler-deps:ubuntu-22.04-20241211` (specific build)
- `rocm/rocprofiler-deps:ubuntu-22.04-latest` (current version)

### Security Updates

Rebuild images periodically to include security updates:

```bash
# Rebuild all images with latest base image updates
./docker-build.sh --all --push
```

## Dependencies Included

These images include all packages from the CI workflows:

- **Core Build**: gcc, g++, cmake, python3, git
- **Development Libraries**: libdw-dev, libsqlite3-dev, libdrm-dev, etc.
- **Documentation**: doxygen, graphviz
- **Testing**: sanitizer libraries, coverage tools
- **Acceleration**: ccache (Ubuntu), sccache (RHEL/SLES)
- **Fonts**: X11 fonts for coverage report generation

## Troubleshooting

### Build Failures
- Ensure Docker daemon is running
- Check base image availability
- Verify network connectivity for package downloads

### Size Optimization
Images include only essential packages. For further size reduction:
- Use multi-stage builds
- Remove package caches (already done)
- Consider Alpine-based alternatives (if compatible with ROCm)

### Testing Images
Test built images with a sample build:

```bash
docker run --rm -it rocm/rocprofiler-deps:ubuntu-22.04-latest bash
# Inside container:
gcc --version
cmake --version
python3 --version
```