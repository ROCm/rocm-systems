# Dockerfile — ROCm dev image + the IREE test tools the rocjitsu-corpus needs.
#
# The corpus regression (scripts/run_gfx1250_regression.sh) drives two IREE
# tools over its packaged gfx1250 VMFBs:
#
#   * iree-check-module       (e2e VMFB checks)
#   * iree-e2e-matmul-test    (matmul VMFB pairs)
#
# Neither ships in the IREE pip wheels (iree-base-runtime only provides
# iree-run-module / iree-benchmark-module / ...): they are *test* tools produced
# only by an IREE source build with -DIREE_BUILD_TESTS=ON. This image builds
# them from source against the HIP HAL driver and stages them onto the standard
# ROCm dev image so mirage can run the corpus inside a container that already
# has everything it needs.
#
# The compiler is intentionally NOT built (-DIREE_BUILD_COMPILER=OFF): the
# corpus runs *prebuilt* VMFBs, so we only need the runtime + the two test
# tools. That keeps the build to the IREE runtime (no LLVM) and the image small.
#
# Build (from this directory):
#   docker build -t mirage/iree-corpus:gfx1250 \
#     --build-arg IREE_REF=v3.7.0 -f Dockerfile .
# or, via mirage's CMake:
#   cmake -S . -B build -DMIRAGE_RUN_CORPUS=ON -DMIRAGE_CORPUS_BUILD_IMAGE=ON
#   cmake --build build --target corpus_image

ARG BASE_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.1.1-complete
FROM ${BASE_IMAGE} AS build

# IREE source ref to build the tools from. Pin a release tag for reproducible
# images; override with --build-arg IREE_REF=<tag|branch|sha>. A recent runtime
# maximises the chance of loading the (new, gfx1250) corpus VMFBs. The packaged
# corpus VMFBs require HAL module version 7, which first shipped during the
# 3.12.0 cycle (the v3.11.0 release only provides version 6).
ARG IREE_REF=iree-3.12.0rc20260604
ARG IREE_REPO=https://github.com/iree-org/iree.git

# Build prerequisites. The ROCm "-complete" base already provides HIP and a
# clang/lld toolchain under /opt/rocm; we add the generic CMake/Ninja/Python
# build deps. Pin nothing here beyond what apt resolves for the base release.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential \
        python3 python3-pip python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Fetch IREE at the requested ref. The compiler is NOT built (below), so the
# giant llvm-project submodule is skipped — only the runtime submodules (incl.
# the HIP build deps) are initialised. Everything is shallow.
RUN git clone --depth 1 --branch "${IREE_REF}" "${IREE_REPO}" /opt/iree-src \
    && git -C /opt/iree-src config submodule.third_party/llvm-project.update none \
    && git -C /opt/iree-src submodule update --init --depth 1 --recommend-shallow --jobs "$(nproc)"

# Configure a runtime-only build with the HIP HAL driver and the test tools.
# IREE's HIP driver dlopen()s libamdhip64.so at run time, so the tools follow
# the host ROCm install mounted into the container at run time. Only the HIP +
# local HAL drivers are enabled (Vulkan/etc. are off) to keep the build deps to
# what the ROCm base already provides.
RUN cmake -G Ninja -S /opt/iree-src -B /opt/iree-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DIREE_BUILD_COMPILER=OFF \
        -DIREE_BUILD_TESTS=ON \
        -DIREE_BUILD_SAMPLES=OFF \
        -DIREE_BUILD_PYTHON_BINDINGS=OFF \
        -DIREE_HAL_DRIVER_DEFAULTS=OFF \
        -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
        -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
        -DIREE_HAL_DRIVER_HIP=ON

# Build just the two tools the corpus invokes (and their deps).
RUN cmake --build /opt/iree-build --target iree-check-module iree-e2e-matmul-test

# ---- final image ------------------------------------------------------------
# Stage the two tools onto a fresh copy of the base image so the result is the
# ROCm dev environment + the IREE test tools, without the multi-GB build tree.
FROM ${BASE_IMAGE} AS runtime

# The tool locations within the build tree are stable across IREE versions, but
# copy by search to be resilient to layout changes.
COPY --from=build /opt/iree-build /opt/iree-build
RUN set -eux; \
    install -d /usr/local/bin; \
    for tool in iree-check-module iree-e2e-matmul-test; do \
        src="$(find /opt/iree-build -type f -name "$tool" -perm -u+x | head -n1)"; \
        test -n "$src" || { echo "error: $tool not found in build tree" >&2; exit 1; }; \
        install -m 0755 "$src" "/usr/local/bin/$tool"; \
    done; \
    rm -rf /opt/iree-build

# Smoke check that the tools are runnable (prints help, exits 0).
RUN iree-check-module --help >/dev/null 2>&1 || true; \
    iree-e2e-matmul-test --help >/dev/null 2>&1 || true

LABEL org.opencontainers.image.title="mirage IREE corpus runner" \
      org.opencontainers.image.description="ROCm dev image + IREE iree-check-module / iree-e2e-matmul-test for the rocjitsu-corpus gfx1250 regression" \
      org.opencontainers.image.base.name="docker.io/rocm/dev-ubuntu-24.04:7.1.1-complete"
