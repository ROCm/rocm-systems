.. meta::
   :description: HSA runtime implementation
   :keywords: ROCR, ROCm, library, tool, runtime

.. _what-is-rocr-runtime:

What is ROCR?
========================

The ROCm runtime (ROCR) is AMD's implementation of HSA runtime, which is a thin, user-mode API that exposes the necessary interfaces to access and interact with graphics hardware driven by the AMDGPU driver set and the ROCK kernel driver. Together they enable you to directly harness the power of discrete AMD graphics devices by allowing host applications to launch compute kernels directly to the graphics hardware.

The ROCR APIs are capable of the following:

- Error handling

- Runtime initialization and shutdown

- System and agent information

- Signals and synchronization

- Architected dispatch

- Memory management

- Fitting into a typical software architecture stack

ROCR provides direct access to the graphics hardware, allowing you more control over execution. An example of low-level hardware access is the support for one or more user-mode queues, which provides a low-latency kernel dispatch interface, allowing you to develop customized dispatch algorithms specific to your application.
The HSA Architected Queuing Language (AQL) is an open standard defined by the HSA Foundation, which specifies the packet syntax used to control supported AMD or ATI Radeon © graphics devices. The AQL language supports several packet types, including packets that can command the hardware to automatically resolve inter-packet dependencies (barrier AND and barrier OR packet), kernel dispatch packets, and agent dispatch packets.
In addition to user-mode queues and AQL, the HSA runtime exposes various virtual address ranges that can be accessed by one or more of the system’s graphics devices and also possibly by the host. The exposed virtual address ranges support either a fine-grained or a coarse-grained access. Updates to memory in a fine-grained region are immediately visible to all devices that can access it, but only one device can have access to a coarse-grained allocation at a time. You can change the ownership of a coarse-grained region using the HSA runtime memory APIs, but this transfer of ownership must be explicitly done by the host application.

For a complete description of the HSA Runtime APIs, AQL, and the HSA memory policy, refer to the `HSA Runtime Programmer’s Reference Manual <https://hsafoundation.com/wp-content/uploads/2021/02/HSA-Runtime-1.2.pdf>`_.
