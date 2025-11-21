FROM ubuntu:22.04

# do not prompt in apt
# https://github.com/moby/moby/issues/4032#issuecomment-163689851
ARG DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NONINTERACTIVE_SEEN=true

# set timezone
ENV TZ="America/Chicago"
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

RUN apt update --yes \
        && apt install --yes \
            build-essential \
            cmake \
            git \
            gnupg \
            libdrm-dev \
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
        && python3 -m pip install --no-cache-dir clang==16.0.1 ctypeslib2==2.3.4 -U \
        && rm -rf $TEMPDIR

# install cmake
RUN TEMPDIR=$(mktemp -d) \
        && cd $TEMPDIR \
        && wget https://github.com/Kitware/CMake/releases/download/v3.29.2/cmake-3.29.2-linux-x86_64.sh && chmod +x cmake*.sh \
        && ./cmake*.sh --skip-license --prefix=/usr \
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
