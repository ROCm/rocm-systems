FROM ubuntu@sha256:4f838adc7181d9039ac795a7d0aba05a9bd9ecd480d294483169c5def983b64d

# do not prompt in apt
# https://github.com/moby/moby/issues/4032#issuecomment-163689851
ARG DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NONINTERACTIVE_SEEN=true

# set timezone
ENV TZ="America/Chicago"
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

RUN useradd -m rocm && USER rocm

RUN apt update --yes \
        && apt install --yes \
            build-essential \
            cmake \
            git \
            gnupg \
            libdrm-dev \
            libmnl-dev \
            libnl-3-dev \
            libnl-genl-3-dev \
            libpython3-dev \
            lsb-release \
            pkg-config \
            pkg-config \
            python3-pip \
            software-properties-common \
            wget \
        && apt clean \
        && rm -rf /var/cache/apt/ /var/lib/apt/lists/*

# install clang
RUN TEMPDIR=$(mktemp -d) \
        && cd $TEMPDIR \
        && wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh \
        && ./llvm.sh 16 \
        && update-alternatives --install /usr/bin/clang clang $(which clang-16) 91 --slave /usr/bin/clang++ clang++ $(which clang++-16) \
        && python3 -m pip install --no-cache-dir clang==16.0.1 ctypeslib2==2.4.0 -U \
        && rm -rf $TEMPDIR

# install cmake
ARG CMAKE_VERSION=3.29.2
RUN TEMPDIR=$(mktemp -d) \
        && cd $TEMPDIR \
        && gpg --keyserver hkps://keyserver.ubuntu.com --recv-keys CBA23971357C2E6590D9EFD3EC8FEF3A7BFB4EDA \
        && wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh \
        && wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-SHA-256.txt \
        && wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-SHA-256.txt.asc \
        && gpg --verify cmake-${CMAKE_VERSION}-SHA-256.txt.asc cmake-${CMAKE_VERSION}-SHA-256.txt \
        && grep "cmake-${CMAKE_VERSION}-linux-x86_64.sh" cmake-${CMAKE_VERSION}-SHA-256.txt | sha256sum -c \
        && chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh \
        && ./cmake-${CMAKE_VERSION}-linux-x86_64.sh --skip-license --prefix=/usr \
        && rm -rf $TEMPDIR

WORKDIR /src
CMD ["sh", "-c", "\
    cp -r /src /tmp/src && \
    cd /tmp/src && \
    rm -rf build .cache && \
    cmake -B build -DBUILD_WRAPPER=ON && \
    make -C build -j $(nproc) && \
    cp /tmp/src/py-interface/amdsmi_wrapper.py /src/py-interface/amdsmi_wrapper.py \
"]
