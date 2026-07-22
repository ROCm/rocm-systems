FROM ubuntu@sha256:4f838adc7181d9039ac795a7d0aba05a9bd9ecd480d294483169c5def983b64d

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
# Installer hash from: curl -sSf https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init.sha256
ARG RUSTUP_INIT_SHA256=4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10
WORKDIR /tmp
RUN curl -sSf -O https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init \
    && echo "${RUSTUP_INIT_SHA256}  rustup-init" | sha256sum -c \
    && chmod +x rustup-init \
    && ./rustup-init -y --default-toolchain stable \
    && rm rustup-init
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /src
