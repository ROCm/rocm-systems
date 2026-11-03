# Local POC: build and trace environment. ROCm comes from the host, mounted read-only.
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends       build-essential cmake ninja-build pkg-config git python3 python3-bt2       lttng-tools liblttng-ust-dev babeltrace2 libnuma-dev libdrm-dev libelf-dev zlib1g-dev       libzstd-dev ca-certificates file  && rm -rf /var/lib/apt/lists/*
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-pip \
 && pip3 install --break-system-packages CppHeaderParser && rm -rf /var/lib/apt/lists/*
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libgl-dev mesa-common-dev libglx-dev \
 && rm -rf /var/lib/apt/lists/*
