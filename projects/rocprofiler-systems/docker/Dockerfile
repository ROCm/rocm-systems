
ARG DISTRO
ARG VERSION
FROM ${DISTRO}:${VERSION}

ENV HOME /root
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US
ENV LC_ALL C
ENV SHELL /bin/bash
ENV BASH_ENV /etc/bash.bashrc
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /tmp
SHELL [ "/bin/bash", "-c" ]

ARG EXTRA_PACKAGES=""
ARG ROCM_REPO_VERSION="debian"

ENV PATH ${HOME}/.local/bin:${PATH}

RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y build-essential cmake libnuma-dev wget gnupg2 m4 bash-completion git-core autoconf libtool autotools-dev python3-pip lsb-release libpapi-dev libpfm4-dev && \
    python3 -m pip install 'cmake==3.18.4' && \
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
    echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/${ROCM_REPO_VERSION}/ ubuntu main" | tee /etc/apt/sources.list.d/rocm.list && \
    apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y rocm-dev rocm-utils roctracer-dev rocprofiler-dev hip-base hsa-amd-aqlprofile hsa-rocr-dev hsakmt-roct-dev ${EXTRA_PACKAGES}

ENV LC_ALL C.UTF-8
WORKDIR /home
SHELL [ "/bin/bash", "--login", "-c" ]
