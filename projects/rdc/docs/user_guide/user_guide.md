# Introduction to ROCm Data Center Tool User Guide

The ROCm™ Data Center Tool™ (RDC) simplifies the administration and addresses key infrastructure challenges in AMD GPUs in cluster and datacenter environments. The main features are:

•        GPU telemetry

•        GPU statistics for jobs

•        Integration with third-party tools

•        Open source

You can use the tool in standalone mode if all components are installed. However, the existing management tools can use the same set of features available in a library format.

For details on different modes of operation, refer to [Starting RDC](install).

## Objective

This user guide is intended to:

•     Provide an overview of the RDC tool features.

•     Describe how system administrators and Data Center (or HPC) users can administer and configure AMD GPUs.

•     Describe the components.

•     Provide an overview of the open source developer handbook.

## Terminology

Table 1: Terminologies and Abbreviations

| Term                     | Description               |
| ------------------------ | ------------------------- |
| RDC                      | ROCm Data Center tool     |
| Compute node (CN)        | One of many nodes containing one or more GPUs in the Data Center on which compute jobs are run |
| Management node (MN) or Main console | A machine running system administration applications to administer and manage the Data Center |
| GPU Groups | Logical grouping of one or more GPUs in a compute node |
| Fields | A metric that can be monitored by the RDC, such as GPU temperature, memory usage, and power usage |
| Field Groups | Logical grouping of multiple fields |
| Job | A workload that is submitted to one or more compute nodes |
 
## Target Audience

The audience for the AMD RDC tool consists of:

•        Administrators: The tool provides the cluster administrator with the capability of monitoring, validating, and configuring policies.

•        HPC Users: Provides GPU-centric feedback for their workload submissions.

•        OEM: Add GPU information to their existing cluster management software.

•        Open source Contributors: RDC is open source and accepts contributions from the community. 
