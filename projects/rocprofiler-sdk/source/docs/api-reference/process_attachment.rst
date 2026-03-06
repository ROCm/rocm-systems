.. meta::
  :description: Technical guide for implementing ROCprofiler-SDK process attachment
  :keywords: ROCprofiler-SDK, rocattach, attach, detach, process attachment, ptrace, dynamic profiling, tool development

.. _process_attachment_implementation:

********************************************************************************
Implementing Process Attachment Tools
********************************************************************************

Overview
========

This document provides the technical details needed to implement a process attachment tool similar to ``rocprofv3 --attach``. Process attachment allows profiling tools to dynamically attach to running GPU applications without requiring application restart. The implementation can use either the provided python or exported C functions.

Direct Python Execution
===================================

The python file ``rocprof-attach`` can be called directly to attach to a specific PID and use custom tools within the attachment target.

.. code-block:: bash

   $ rocprof-attach -p 12345 -t path/to/your-tool-library.so -d 5000

In this example, the process with PID 12345 will be attached to and the library path/to/your-tool-library.so will be loaded by rocprofiler-sdk from within that process. After 5000 milliseconds have passed, detach will be called and ``rocprof-attach`` will exit when detachment is complete.

More information can be found by invoking ``rocprof-attach -h``


Python Functions
===================================

The python file ``rocprof-attach`` defines an attach function that can be used for attachment:

.. code-block:: python

   def attach(
     pid,
     attach_tool_library,
     attach_duration_msec,
     attach_library=ROCPROF_ATTACH_LIBRARY,
   ):

**Function Details**

The attach function performs the entire attachment process, including attaching and detaching, and provides the ability to use custom tools via the tool_libraries parameter. It also has simple control flow intended for direct calling from python. For more complex control, it is recommended to instead use the explicit attach and detach functions provided by the ``librocprofiler-sdk-rocattach.so`` binary.

**Parameters**

- **pid**: Required - PID of process to attach to
   - Defaults to environment variable ROCPROF_ATTACH_PID
- **attach_tool_library**: Colon delimited list of tool libraries to use
   - Defaults to environment variable ROCPROF_ATTACH_TOOL_LIBRARY
- **attach_duration_msec**: Optional - Length of time in milliseconds to profile for
   - Defaults to environment variable ROCPROF_ATTACH_DURATION
   - If unspecified, attachment will run until Enter is pressed or SIGINT (Ctrl+C) is received
- **attach_library**: Optional - Tool library to use for attachment and detachment
   - Default will work for nearly all applications
   - Defaults to environment variable ROCPROF_ATTACH_LIBRARY
   - If unspecified, defaults to the absolute path of librocprofiler-sdk-rocattach.so

C Functions
===================================

The C library ``librocprofiler-sdk-rocattach.so`` defines an attach and detach function that can be used for attachment:

.. code-block:: cpp

   extern "C" {
       // Start attachment to a target process
       rocattach_status_t rocattach_attach(int pid) ROCATTACH_API;

       // Detach from target process and cleanup
       rocattach_status_t rocattach_detach(int pid) ROCATTACH_API;
   }

**Function Details:**

- **rocattach_attach(int pid)**: Main entry point for starting attachment to a process
   - Takes the target process ID as parameter
   - Initiates ptrace-based attachment sequence
   - Custom tool libraries can be specified in a colon delimited list with the environment variable ROCPROF_ATTACH_TOOL_LIBRARY

- **rocattach_detach(int pid)**: Entry point for detaching from the target process
   - Takes the target process ID as a parameter
   - Cleans up attachment resources and terminates profiling
   - A PID of 0 can be specified to detach from all processes

Function Call Sequence
======================

Initial Attachment Sequence
---------------------------

The initial attachment process roughly follows this sequence:

1. rocattach_attach(pid) ← Your tool calls this
2. ptrace calls rocprofiler_register_attach(env_buffer)
3. tool_library::rocprofiler_configure(...)
4. tool_library::rocprofiler_configure_attach(...)
5. tool_library::tool_init(...)
6. tool_library::tool_attach(...)
7. [Profiling and data collection...]
8. rocattach_detach(pid) ← Your tool calls this
9. ptrace calls rocprofiler_register_detach()
10. tool_library::tool_detach(...)
11. [Program ends]   
12. tool_library::tool_fini(...)

Reattachment Sequence
---------------------

For reattachment to a previously attached process:

1. rocattach_attach(pid) ← Your tool calls this again
2. ptrace calls rocprofiler_register_attach(env_buffer)
3. tool_library::tool_attach(...)
4. [Continued profiling and data collection...]
5. rocattach_detach(pid) ← Your tool calls this
6. ptrace calls rocprofiler_register_detach()
7. tool_library::tool_detach(...)


Environment Variable Configuration
==================================

The target process must have ``ROCP_TOOL_ATTACH=1`` set, or be using a version of ``rocprofiler-register`` configured with the CMake flag ``ROCPROFILER_REGISTER_BUILD_DEFAULT_ATTACHMENT=ON``

Required Variables
------------------

.. code-block:: text

   export ROCP_TOOL_ATTACH=1
   OR
   cmake /path/to/rocprofiler-register -DROCPROFILER_REGISTER_BUILD_DEFAULT_ATTACHMENT=ON

Tool Library Configuration
--------------------------

The attachment system can use any tool library. ``librocprofiler-sdk-tool.so`` is used when the environment variable is not set.

.. code-block:: cpp

   // Attachment libraries to be used
   setenv("ROCPROF_ATTACH_TOOL_LIBRARY", "example-tool-1.so:example-tool-2.so", 1);

Using the Attachment Functions
==============================

This is a simplified example of how to use these functions in your own attachment tool:

Basic Attachment Implementation
-------------------------------

.. code-block:: cpp

   #include <rocattach.h>

   #include <dlfcn.h>
   #include <iostream>
   #include <thread>
   #include <chrono>

   class ROCprofilerAttachmentTool {
   private:
       void* attach_lib_handle = nullptr;
       rocattach_status_t (*attach_func)(int) = nullptr;
       rocattach_status_t (*detach_func)(int) = nullptr;

   public:
       bool initialize() {
           // Load the rocprofiler-attach library/binary
           attach_lib_handle = dlopen("librocprofiler-sdk-rocattach.so", RTLD_NOW);
           if (!attach_lib_handle) {
               std::cerr << "Failed to load librocprofiler-sdk-rocattach: " << dlerror() << std::endl;
               return false;
           }

           // Get the attachment function pointers
           attach_func = (rocattach_status_t(*)(int))dlsym(attach_lib_handle, "rocattach_attach");
           detach_func = (rocattach_status_t(*)(int))dlsym(attach_lib_handle, "rocattach_detach");

           if (!attach_func || !detach_func) {
               std::cerr << "Failed to find attachment functions" << std::endl;
               return false;
           }

           return true;
       }

       bool attach_to_process(pid_t pid, uint32_t duration_ms) {
           // Validate the target process
           if (kill(pid, 0) != 0) {
               std::cerr << "Target process " << pid << " is not accessible" << std::endl;
               return false;
           }

           std::cout << "Attaching to process " << pid << std::endl;

           // Start attachment - this will handle all ptrace operations
           if (!attach_func(pid))
           {
               return false;
           }

           // Profile for specified duration
           std::cout << "Profiling for " << duration_ms << " milliseconds..." << std::endl;
           std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

           // Stop profiling
           if (!detach_func(pid))
           {
               return false;
           }

           std::cout << "Profiling completed" << std::endl;
           return true;
       }

       ~ROCprofilerAttachmentTool() {
           if (attach_lib_handle) {
               dlclose(attach_lib_handle);
           }
       }
   };

Main Implementation
-------------------

.. code-block:: cpp

   #include <iostream>
   #include <vector>
   #include <string>
   #include <cstdlib>

   int main(int argc, char* argv[]) {
       if (argc < 2) {
           std::cerr << "Usage: " << argv[0] << " <PID> [duration_ms]" << std::endl;
           std::cerr << "  PID: Process ID to attach to" << std::endl;
           std::cerr << "  duration_ms: Optional profiling duration in milliseconds" << std::endl;
           return 1;
       }

       pid_t target_pid = std::stoi(argv[1]);
       uint32_t duration = (argc > 2) ? std::stoi(argv[2]) : 1000;

       // For this example, the tool library "librocprofiler-sdk-tool.so" is used by
       // default because ROCPROF_ATTACH_TOOL_LIBRARY is not set. These environment
       // variables are used to communicate profiling options to rocprofiler-sdk-tool.
       setenv("ROCPROF_HIP_RUNTIME_API_TRACE", "1", 1);
       setenv("ROCPROF_KERNEL_TRACE", "1", 1);
       setenv("ROCPROF_MEMORY_COPY_TRACE", "1", 1);
       setenv("ROCPROF_OUTPUT_PATH", "./attachment-output", 1);
       setenv("ROCPROF_OUTPUT_FILE_NAME", "attached_profile", 1);

       // Initialize and run attachment tool
       ROCprofilerAttachmentTool tool;
       if (!tool.initialize()) {
           std::cerr << "Failed to initialize attachment tool" << std::endl;
           return 1;
       }

       if (!tool.attach_to_process(target_pid, duration)) {
           std::cerr << "Attachment failed" << std::endl;
           return 1;
       }

       std::cout << "Attachment completed successfully" << std::endl;
       return 0;
   }
