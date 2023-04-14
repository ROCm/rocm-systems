FROM --platform=linux/amd64 rockylinux:8

RUN yum -y install epel-release
RUN yum -y install dnf-plugins-core
RUN yum -y install cmake
RUN yum -y install gcc-c++
RUN yum -y install python39

# Enable rocm package repo
COPY rocm.repo /etc/yum.repos.d
RUN yum config-manager --set-enabled powertools

# Desired ROCm packages
RUN yum -y install rocm-dev5.1.3