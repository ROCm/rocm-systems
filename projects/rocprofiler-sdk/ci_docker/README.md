# ROCProfiler-SDK CI Dependency Docker Images

This directory contains a multi-stage flow for building ROCm CI base images for `rocprofiler-sdk`:

- Stage 1 (Dependencies): OS-specific toolchains, libs, Python reqs, repo clone + submodules, submodule cache.
- Stage 2 (ROCm Base): Configures AMDGPU/ROCm repos, installs core ROCm packages, fetches/extracts therock tarball into `/opt/rocm`.
- Stage 3 (rocm-systems deps): Builds `projects/rocprofiler-register`, `projects/rocr-runtime`, `projects/aqlprofile` from source, plus optional media dependencies `rocDecode` and `rocJPEG` (all toggleable).

## Images Available

### Ubuntu 22.04 (`Dockerfile.ubuntu-22.04`)
Base: `docker.io/rocm/rocprofiler-private:ubuntu-22.04`

### Ubuntu 24.04 (`Dockerfile.ubuntu-24.04`)
Base: `docker.io/rocm/rocprofiler-private:ubuntu-24.04`

### AlmaLinux 8.10 (`Dockerfile.almalinux-8.10`)
Base: `docker.io/rocm/rocprofiler-private:almalinux-8.10`

### RHEL 10 (`Dockerfile.almalinux-10`)
Base: `docker.io/rocm/rocprofiler-private:almalinux-10`

### SLES 15.7 (`Dockerfile.sles-15.7`)
Base: `docker.io/rocm/rocprofiler-private:sles-15.7`

All stage-1 images include:
- Build tools (gcc toolchains, clang, cmake), development libraries (elfutils, sqlite, libdrm, etc.)
- Python (pip/venv) and documentation tools (doxygen, graphviz)
- Acceleration (ccache on Ubuntu, sccache on RHEL/SLES)
- Minimal repo sparse-checkout of `projects/rocprofiler-sdk/requirements.txt` only, then cleanup

## Tagging Convention

- Stage 1 (OS-only):
  - `docker.io/rocm/rocprofiler-private:<OS>-<VER>-<DATE>`
  - `docker.io/rocm/rocprofiler-private:<OS>-<VER>-latest`

- Stage 2 (OS + GPU tarball):
  - `docker.io/rocm/rocprofiler-private:<OS>-<VER>-<GPU>-<DATE>`
  - `docker.io/rocm/rocprofiler-private:<OS>-<VER>-<GPU>-latest`

Examples:
- `docker.io/rocm/rocprofiler-private:ubuntu-22.04-gfx94X-latest`
- `docker.io/rocm/rocprofiler-private:ubuntu-24.04-gfx94X-latest`
- `docker.io/rocm/rocprofiler-private:almalinux-10-gfx110X-20250115`

## Building Images

### Using the Build Script (Recommended)

```bash
# Build all distributions (Ubuntu 22.04/24.04, AlmaLinux 8.10, RHEL 10, SLES 15.7) and all GPUs
projects/rocprofiler-sdk/ci_docker/docker-build.sh --all

# Build specific distributions
projects/rocprofiler-sdk/ci_docker/docker-build.sh ubuntu-22.04 almalinux-10

# Limit GPUs (choose any of: gfx94X,gfx950,gfx110X,gfx120X)
projects/rocprofiler-sdk/ci_docker/docker-build.sh ubuntu-22.04 --gpus gfx94X

# Build and push to registry
projects/rocprofiler-sdk/ci_docker/docker-build.sh --push ubuntu-22.04 ubuntu-24.04

# Show help
projects/rocprofiler-sdk/ci_docker/docker-build.sh --help
```

#### CLI options

- **-h, --help**: Show help
- **-p, --push**: Push images to registry after building
- **-a, --all**: Build all distributions (default)
- **-g, --gpus <list>**: Comma-separated GPU list; default builds all. Example: `--gpus gfx94X,gfx950`
- **--skip-rocm**: Build only Stage 1 (skip Stages 2–4)

Distributions you can pass positionally (one or more):
- **ubuntu-22.04** (22.04)
- **ubuntu-24.04** (24.04)
- **almalinux-8.10** (8.10)
- **almalinux-10** (10.0)
- **sles-15.7** (15.7)

Examples:

```bash
# Stage 1 only (skip Stage 2/3/4) for all distributions
projects/rocprofiler-sdk/ci_docker/docker-build.sh --skip-rocm --all

# Build Ubuntu 24.04 with a single GPU target
projects/rocprofiler-sdk/ci_docker/docker-build.sh ubuntu-24.04 --gpus gfx94X
```

The script will:
- Resolve the latest "therock" tarball per GPU from S3 (no-sign-request)
- Build Stage 1 OS images
- Build Stage 2 OS+GPU images that download and extract the tarball to `/opt/rocm`
- Optionally build Stage 3 (rocm-systems deps including media dependencies)
- Optionally push all produced tags


### Manual Building (advanced)

Stage 1:
```bash
docker build -f projects/rocprofiler-sdk/ci_docker/Dockerfile.ubuntu-22.04 \
  -t docker.io/rocm/rocprofiler-private:ubuntu-22.04-latest \
  projects/rocprofiler-sdk/ci_docker
```

Stage 2 (requires a tarball key):
```bash
docker build -f projects/rocprofiler-sdk/ci_docker/Dockerfile.stages \
  --target stage2 \
  --build-arg BASE_IMAGE=docker.io/rocm/rocprofiler-private:ubuntu-22.04-latest \
  --build-arg GPU_TYPE=gfx94X \
  --build-arg TARBALL_KEY=therock-dist-linux-gfx94X-...tar.gz \
  -t docker.io/rocm/rocprofiler-private:ubuntu-22.04-gfx94X-latest \
  projects/rocprofiler-sdk/ci_docker

Stage 3 (optional rocm-systems deps including media dependencies):
```bash
docker build -f projects/rocprofiler-sdk/ci_docker/Dockerfile.stages \
  --target stage3 \
  --build-arg BASE_IMAGE=docker.io/rocm/rocprofiler-private:ubuntu-22.04-gfx94X-latest \
  --build-arg BUILD_ROCR_RUNTIME=false \
  --build-arg BUILD_ROCDECODE=false \
  --build-arg BUILD_ROCJPEG=false \
  -t docker.io/rocm/rocprofiler-private:ubuntu-22.04-gfx94X-stage3-latest \
  projects/rocprofiler-sdk/ci_docker
```
```

## Using in CI

Reference the OS+GPU tag (stage-2) in your workflow:

```yaml
container:
  image: docker.io/rocm/rocprofiler-private:ubuntu-22.04-gfx94X-latest
  credentials:
    username: ${{ secrets.ROCPROFILER_AZURE_CI_USER }}
    password: ${{ secrets.ROCPROFILER_AZURE_CI_PASS }}
```

### Submodule cache bundled in the image

Stage-1 images include a prebuilt archive of the repository's submodules at `/opt/rocprofiler-submodules-cache.tar.gz` and a helper script `/usr/local/bin/restore-submodules-cache.sh` to restore them into your workspace. This reduces network fetches during `git submodule update`.

Usage in a GitHub Actions job (inside the container):

```yaml
jobs:
  build:
    runs-on: rocprofiler-navi3-dind
    container:
      image: docker.io/rocm/rocprofiler-private:ubuntu-22.04-gfx94X-latest
      credentials:
        username: ${{ secrets.ROCPROFILER_AZURE_CI_USER }}
        password: ${{ secrets.ROCPROFILER_AZURE_CI_PASS }}
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          submodules: false
          set-safe-directory: true

      - name: Restore submodules from image cache
        shell: bash
        run: |
          /usr/local/bin/restore-submodules-cache.sh "$GITHUB_WORKSPACE"
          git config --global --add safe.directory '*'

      - name: Init/Update submodules
        shell: bash
        run: git submodule update --init --recursive --jobs 16
```

Notes:
- The restore step should run after checkout so that `$GITHUB_WORKSPACE/.git` exists.
- The helper script defaults to `/github/workspace` if no path is provided; passing `$GITHUB_WORKSPACE` is recommended.
- You can keep your existing submodule cache logic as a fallback; this image-level cache works even without network access to submodule remotes.

### Cache-busting for repository changes

Stage-1 uses build args to control checkout so Docker caching is deterministic:

- `ROCM_SYSTEMS_REF` (default: `develop`)
- `ROCM_SYSTEMS_REV` (optional commit SHA; set to empty to use the ref tip)

Examples:

```bash
# Pin to a specific commit
ROCM_SYSTEMS_REV=<sha> projects/rocprofiler-sdk/ci_docker/docker-build.sh ubuntu

# Force cache bust by changing the rev value
ROCM_SYSTEMS_REV=$(date +%s) projects/rocprofiler-sdk/ci_docker/docker-build.sh ubuntu
```

Stage-4 similarly supports cache-busting build args for external repos:

- `ROCDECODE_REF`/`ROCDECODE_REV`
- `ROCJPEG_REF`/`ROCJPEG_REV`

## Maintenance

### Updating Dependencies

When CI requirements change:

1. Update the appropriate Dockerfile(s)
2. Rebuild and test the images locally
3. Push updated images to the registry
4. Update CI workflows to use new image tags

### Version Management

Images are tagged with both date and "latest":
- `docker.io/rocm/rocprofiler-private:ubuntu-22.04-20250115` (stage-1 specific build)
- `docker.io/rocm/rocprofiler-private:ubuntu-22.04-latest` (stage-1 current)
- `docker.io/rocm/rocprofiler-private:ubuntu-22.04-gfx94X-20250115` (stage-2 specific build)
- `docker.io/rocm/rocprofiler-private:ubuntu-22.04-gfx94X-latest` (stage-2 current)

### Security Updates

Rebuild images periodically to include security updates:

```bash
# Rebuild all images with latest base image updates
projects/rocprofiler-sdk/ci_docker/docker-build.sh --all --push
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
docker run --rm -it docker.io/rocm/rocprofiler-private:ubuntu-22.04-gfx94X-latest bash
# Inside container:
gcc --version
cmake --version
python3 --version
```