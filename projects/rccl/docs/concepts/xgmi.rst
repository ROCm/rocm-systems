.. meta::
   :description: How to use RCCL
   :keywords: RCCL, ROCm, library, API,

.. _xgmi:

****************************
xGMI RCCL integration
****************************

xGMI (External Global Memory Interconnect) is AMD’s high-speed GPU-to-GPU interconnect based on Infinity Fabric™ technology. 
It is designed to enable efficient peer-to-peer GPU communication for multi-GPU AI and HPC workloads. 
This interconnect is crucial for ensuring that large-scale AI models and distributed HPC simulations run efficiently across multiple GPUs. 
Each AMD Instinct MI300X system GPU is connected to its seven peer GPUs via xGMI links, forming a fully connected mesh with high-bandwidth, low-latency communication. 
For more information related to the MI300X Platform, refer to this datasheet.
