FROM --platform=linux/amd64 registry.suse.com/suse/sle15:15.3

RUN zypper -n --gpg-auto-import-keys addrepo https://download.opensuse.org/repositories/devel:/languages:/perl/SLE_15_SP3/devel:languages:perl.repo 

COPY amdgpu.repo /etc/zypp/repos.d/amdgpu.repo

RUN zypper -n --gpg-auto-import-keys install rocm-dev5.1.3

RUN zypper -n install gcc-c++ && \
    zypper -n install make && \
    zypper -n install git
    