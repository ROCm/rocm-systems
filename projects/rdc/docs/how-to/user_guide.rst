.. meta::
  :description: documentation of the installation, configuration, and use of the ROCm Data Center tool
  :keywords: ROCm Data Center tool, RDC, ROCm, API, reference, data type, support

.. _rdc-use:

******************************************
Introduction to the RDC tool
******************************************

The ROCm Data Center tool (RDC) simplifies the administration and addresses key infrastructure challenges in AMD GPUs in cluster and datacenter environments. The main features are:

* GPU telemetry
* GPU statistics for jobs
* Integration with third-party tools
* Open source

You can use the RDC tool in standalone mode if all components are installed. However, the existing management tools can use the same set of features available in a library format.

For details on different modes of operation, refer to *Starting RDC* in :ref:`rdc-install`.

Target Audience
===============

The audience for the AMD RDC tool consists of:

* Administrators: RDC provides the cluster administrator with the capability of monitoring, validating, and configuring policies.
* HPC Users: Provides GPU-centric feedback for their workload submissions.
* OEM: Add GPU information to their existing cluster management software.
* Open source Contributors: RDC is open source and accepts contributions from the community.

Objective
=========

This documentation will:

#. Introduce the tool features in :ref:`rdc-features`
#. Describe integration with external tools in :ref:`rdc-3rd-party`
#. Provide an open source handbook in :ref:`rdc-handbook`
#. Introduce elements of the tool API in :ref:`api-intro`

Terminology
===========

.. list-table:: Terminologies and Abbreviations

    * - **Terms**
      - **Description**

    * - RDC
      - ROCm Data Center tool

    * - Compute node (CN)
      - One of many nodes containing one or more GPUs in the Data Center on which compute jobs are run

    * - Management node (MN) or Main console
      - A machine running system administration applications to administer and manage the Data Center

    * - GPU Groups
      - Logical grouping of one or more GPUs in a compute node

    * - Fields
      - A metric that can be monitored by the RDC, such as GPU temperature, memory usage, and power usage

    * - Field Groups
      - Logical grouping of multiple fields

    * - Job
      - A workload that is submitted to one or more compute nodes


