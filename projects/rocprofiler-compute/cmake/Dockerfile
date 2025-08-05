##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

FROM ubuntu:20.04

USER root

COPY rocm_install.sh /rocprofiler-compute/rocm_install.sh

ENV PATH="/rocprofiler-compute:${PATH}"
ENV TZ="US/Chicago"
ENV DEBIAN_FRONTEND noninteractive

#pyenv dependencies
RUN apt update && \
	apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl
ENV HOME="/rocprofiler-compute"
WORKDIR $HOME
ENV  PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
	apt update  && \
	apt-get install -y cmake wget git python3-dev rpm python3-venv  software-properties-common &&\
	add-apt-repository ppa:deadsnakes/ppa -y &&\
	apt install python3.7 -y libpython3.7-dev python3.7-venv libnuma-dev &&\
	curl https://pyenv.run | bash

RUN	echo "export PATH=$HOME/.pyenv/bin:$PATH" >> ~/.bashrc &&\
	echo eval "$(pyenv init -)" >> ~/.bashrc &&\
	echo eval "$(pyenv virtualenv-init -)" >> ~/.bashrc &&\
    CPPFLAGS=-I/usr/bin/openssl \
    LDFLAGS=-L/usr/lib64 \
    CONFIGURE_OPTS=--enable-shared pyenv install -v 3.8.12 &&\
	pyenv global 3.8.12 &&\
	apt-get install -y python3-pip
	#clang?

RUN python3 -m pip install astunparse==1.6.2 colorlover dash matplotlib numpy pandas pymongo pyyaml tabulate tqdm dash-svg pyinstaller dash-bootstrap-components &&\
	python3 -m pip install 'cmake==3.21.4' && \
	./rocm_install.sh &&\
    #wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
    #echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/${ROCM_REPO_VERSION}/ ${ROCM_REPO_DIST} main" | tee /etc/apt/sources.list.d/rocm.list && \
    apt-get update && \
    apt-get dist-upgrade -y && \
    #apt-get install -y rocm-dev rocm-utils rocm-smi-lib roctracer-dev rocprofiler-dev rccl-dev hip-base hsa-amd-aqlprofile hsa-rocr-dev hsakmt-roct-dev ${EXTRA_PACKAGES} && \
    apt-get autoclean
