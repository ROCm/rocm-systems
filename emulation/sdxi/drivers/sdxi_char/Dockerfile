# Build-test sdxi_char against Ubuntu mainline kernel headers.
# Verifies that the driver compiles with CONFIG_IO_URING=y, catching the
# class of errors (licence, declaration-after-statement, bad API calls)
# that only surface when the full header set is present.
#
# Usage:
#   # Pre-download the kernel header debs on the host (wget works; Docker
#   # container can't reach kernel.ubuntu.com through the ZScaler proxy):
#   BASE=https://kernel.ubuntu.com/mainline/v7.2.6/amd64
#   wget -P emulation/sdxi/drivers/sdxi_char \
#       "$BASE/linux-headers-7.2.6-070206_7.2.6-070206.202609141300_all.deb" \
#       "$BASE/linux-headers-7.2.6-070206-generic_7.2.6-070206.202609141300_amd64.deb"
#   docker build -t sdxi-char-build emulation/sdxi/drivers/sdxi_char

FROM ubuntu:24.04

ARG KERNEL=7.2.6-070206
ARG KERNEL_BUILD=7.2.6-070206.202609141300
RUN apt-get update && apt-get install -y --no-install-recommends \
        bc \
        bison \
        build-essential \
        flex \
        gcc-14 \
        kmod \
        libdw-dev \
        libelf-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install kernel headers pre-downloaded on the host to avoid proxy SSL issues.
COPY linux-headers-${KERNEL}_${KERNEL_BUILD}_all.deb \
     linux-headers-${KERNEL}-generic_${KERNEL_BUILD}_amd64.deb /tmp/
RUN dpkg -i \
        "/tmp/linux-headers-${KERNEL}_${KERNEL_BUILD}_all.deb" \
        "/tmp/linux-headers-${KERNEL}-generic_${KERNEL_BUILD}_amd64.deb" \
    && rm /tmp/*.deb

COPY Makefile sdxi_char.c sdxi_user.h /src/

RUN make -C /usr/src/linux-headers-${KERNEL}-generic M=/src CC=gcc-14 modules \
    && echo "sdxi_char.ko built successfully" \
    && ls -lh /src/sdxi_char.ko
