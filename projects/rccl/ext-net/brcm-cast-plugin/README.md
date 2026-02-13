# Broadcom RCCL CAST Network Plugin

This document describes a RCCL Network Plugin that achieves congestion aware
traffic distribution using standard RoCEv2.

## Overview

The RCCL Network Plugin supports Broadcom's Congestion Aware Sprayed Traffic
(CAST) feature, which is an implementation of QP Load Balance (LB) Scheduling
that provides performance improvements over typical QP Load Balancing. The
idea is to use Round Trip Time (RTT) measurements to schedule more traffic
over the QPs that are experiencing the least congestion. 

## Installing and Building the CAST Network Plugin

This document describes in-tree installation, where the CAST Network Plugin
is installed in the ext-net area of the RCCL tree.

The following description assumes the RCCL_HOME variable is set to the
path to the root of the RCCL tree. For example:

export RCCL_HOME=/rccl/rccl-rocm-7.0.2

To install the CAST Network Plugin:

cd $RCCL_HOME/ext-net/brcm-cast

To build the CAST Network Plugin:

make

To copy the CAST Network Plugin to the RCCL tree build directory:

cp librccl-net-bnxt.so ../../../build/librccl-net.so

## Run rccl with CAST Net plugin
An example RCCL command line that invokes the CAST Network Plugin is shown 
below:

export RCCL_HOME=/rccl/rccl-rocm-7.0.2; 
/root/benchmarks_build/ubuntu_22.04/openmpi_4.1.6_ucx_1.15.0/install/bin/mpirun --mca orte_base_help_aggregate 0 -np 32 -host 1.1.70.15:8,1.1.72.15:8,1.1.14.15:8,1.1.16.15:8 --allow-run-as-root --gmca btl_tcp_if_include eno8303 --gmca oob_tcp_if_include eno8303 --gmca btl tcp,self -x NCCL_IB_DISABLE=0 -x NCCL_IB_HCA=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re6:1,bnxt_re7:1 -x NCCL_IB_TC=104 -x NCCL_IB_GID_INDEX=3 -x NCCL_IGNORE_CPU_AFFINITY=1 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_IB_QP_SCHED_ENABLE=1 -x NCCL_IB_SPLIT_DATA_ON_QPS=1 -x NCCL_DEBUG=VERSION -x LD_LIBRARY_PATH=$RCCL_HOME/build /root/rccl/rccl-tests_7.0.2/build/all_gather_perf -b 1M -e 16G -f 2 -n 20 -w 5 -g 1 -c 1 -p 1 -t 1 

## Network Plugin API Version

The CAST Plugin is implemented using the Network Transport Plugin API v10.


