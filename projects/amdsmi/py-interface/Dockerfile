FROM ubuntu@sha256:4f838adc7181d9039ac795a7d0aba05a9bd9ecd480d294483169c5def983b64d

# do not prompt in apt
# https://github.com/moby/moby/issues/4032#issuecomment-163689851
ARG DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NONINTERACTIVE_SEEN=true

# set timezone
ENV TZ="America/Chicago"
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

RUN useradd -m rocm

RUN apt update --yes \
        && apt install --yes \
            build-essential \
            ca-certificates \
            cmake \
            git \
            gnupg \
            libdrm-dev \
            libmnl-dev \
            libnl-3-dev \
            libnl-genl-3-dev \
            libpython3-dev \
            libssl-dev \
            lsb-release \
            pkg-config \
            python3-pip \
            software-properties-common \
            wget \
        && apt clean \
        && rm -rf /var/cache/apt/ /var/lib/apt/lists/*

# install clang-16 from apt.llvm.org
# Acquire::https peer verification is disabled for apt.llvm.org to account for
# proxy intercepting TLS and presenting untrusted cert.
# Package integrity guaranteed by the GPG signature on the repo metadata.
RUN echo 'Acquire::https::apt.llvm.org::Verify-Peer "false";' > /etc/apt/apt.conf.d/99llvm-no-verify \
        && apt update --yes \
        && apt install --yes apt-transport-https \
        && apt clean \
        && rm -rf /var/cache/apt/ /var/lib/apt/lists/* \
        && wget --no-check-certificate -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor | tee /etc/apt/trusted.gpg.d/apt.llvm.org.gpg > /dev/null \
        && echo "deb https://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main" > /etc/apt/sources.list.d/llvm-16.list \
        && apt update --yes \
        && apt install --yes clang-16 \
        && apt clean \
        && rm -rf /var/cache/apt/ /var/lib/apt/lists/* \
        && update-alternatives --install /usr/bin/clang clang $(which clang-16) 91 --slave /usr/bin/clang++ clang++ $(which clang++-16) \
        && python3 -m pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org clang==16.0.1 ctypeslib2==2.4.0 -U

WORKDIR /src
CMD ["sh", "-c", "\
    cp -r /src /tmp/src && \
    cd /tmp/src && \
    rm -rf build .cache && \
    cmake -B build -DBUILD_WRAPPER=ON && \
    make -C build -j $(nproc) && \
    cp /tmp/src/py-interface/amdsmi_wrapper.py /src/py-interface/amdsmi_wrapper.py \
"]
