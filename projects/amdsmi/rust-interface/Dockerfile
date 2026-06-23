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
            curl \
            git \
            libclang-dev \
            libdrm-dev \
            libmnl-dev \
            libnl-3-dev \
            libnl-genl-3-dev \
            pkg-config \
        && apt clean \
        && rm -rf /var/cache/apt/ /var/lib/apt/lists/*

# install Rust toolchain
RUN curl -sSf -o /tmp/rustup-init https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init \
    && curl -sSf -o /tmp/rustup-init.sha256 https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init.sha256 \
    && sha256sum -c /tmp/rustup-init.sha256 \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --default-toolchain stable \
    && rm /tmp/rustup-init /tmp/rustup-init.sha256
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /src
