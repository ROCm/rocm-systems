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

Python Functions
===================================

The python file ``rocprofv3-attach.py`` defines a main function that can be used for attachment:

.. code-block:: python

   def main(
       pid=os.environ.get("ROCPROF_ATTACH_PID", None),
       attach_library=os.environ.get(
           "ROCPROF_ATTACH_TOOL_LIBRARY", ROCPROF_ATTACH_TOOL_LIBRARY
       ),
       duration=os.environ.get("ROCPROF_ATTACH_DURATION", None),
   )

**Function Details**

The main function performs the entire attachment process, including attaching and detaching, and provides the ability to use custom tools. It also has simple control flow intended for direct calling from python. For more complex control, it is recommended to instead use the explicit attach and detach functions provided by the ``librocprofv3-attach.so`` binary.

**Parameters**

- **pid**: Required - PID of process to attach to
   - Defaults to environment variable ROCPROF_ATTACH_PID
- **attach_library**: Optional - Colon delimited list of tool libraries to use
   - Defaults to environment variable ROCPROF_ATTACH_TOOL_LIBRARY
   - If unspecified, defaults to the absolute path of librocprofv3-attach.so
- **duration**: Optional - Length of time in milliseconds to profile for
   - Defaults to environment variable ROCPROF_ATTACH_DURATION
   - If unspecified, attachment will run until Enter is pressed or SIGINT (Ctrl+C) is received

C Functions
===================================

The C library ``librocprofv3-attach.so`` defines an attach and detach function that can be used for attachment:

.. code-block:: cpp

   extern "C" {
       // Start attachment to a target process
       void attach(uint32_t pid) ROCPROFILER_EXPORT;

       // Detach from target process and cleanup
       void detach() ROCPROFILER_EXPORT;
   }

**Function Details:**

- **attach(uint32_t pid)**: Main entry point for starting attachment to a process
   - Takes the target process ID as parameter
   - Initiates ptrace-based attachment sequence
   - Custom tool libraries can be specified in a colon delimited list with the environment variable ROCPROF_ATTACH_TOOL_LIBRARY

- **detach()**: Entry point for detaching from the target process
   - Cleans up attachment resources and terminates profiling

Function Call Sequence
======================

Initial Attachment Sequence
---------------------------

The initial attachment process roughly follows this sequence:

1. attach(pid) ← Your tool calls this
2. ptrace calls rocprofiler_register_attach(env_buffer)
3. tool_library::rocprofiler_configure(...)
4. tool_library::rocprofiler_configure_attach(...)
5. tool_library::tool_init(...)
6. tool_library::tool_attach(...)
7. [Profiling and data collection...]
8. detach() ← Your tool calls this
9. ptrace calls rocprofiler_register_detach()
10. tool_library::tool_detach(...)
11. [Program ends]   
12. tool_library::tool_fini(...)

Reattachment Sequence
---------------------

For reattachment to a previously attached process:

1. attach(pid) ← Your tool calls this again
2. ptrace calls rocprofiler_register_attach(env_buffer)
3. tool_library::tool_attach(...)
4. [Continued profiling and data collection...]
5. detach() ← Your tool calls this
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

Here's how to use these functions in your own attachment tool:

Basic Attachment Tool Implementation
------------------------------------

.. code-block:: cpp

   #include <dlfcn.h>
   #include <iostream>
   #include <thread>
   #include <chrono>

   class ROCprofilerAttachmentTool {
   private:
       void* attach_lib_handle = nullptr;
       void (*attach_func)(uint32_t) = nullptr;
       void (*detach_func)() = nullptr;

   public:
       bool initialize() {
           // Load the rocprofiler-attach library/binary
           attach_lib_handle = dlopen("librocprofv3-attach.so", RTLD_NOW);
           if (!attach_lib_handle) {
               std::cerr << "Failed to load librocprofv3-attach: " << dlerror() << std::endl;
               return false;
           }

           // Get the attachment function pointers
           attach_func = (void(*)(uint32_t))dlsym(attach_lib_handle, "attach");
           detach_func = (void(*)())dlsym(attach_lib_handle, "detach");

           if (!attach_func || !detach_func) {
               std::cerr << "Failed to find attachment functions" << std::endl;
               return false;
           }

           return true;
       }

       bool attach_to_process(pid_t pid, uint32_t duration_ms = 0) {
           // Validate the target process
           if (kill(pid, 0) != 0) {
               std::cerr << "Target process " << pid << " is not accessible" << std::endl;
               return false;
           }

           std::cout << "Attaching to process " << pid << std::endl;

           // Start attachment - this will handle all ptrace operations
           attach_func(pid);

           if (duration_ms > 0) {
               // Profile for specified duration
               std::cout << "Profiling for " << duration_ms << " milliseconds..." << std::endl;
               std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

               // Stop profiling
               detach_func();
           } else {
               std::cout << "Profiling until process ends or manual detach..." << std::endl;
               // Monitor process or wait for external signal to detach
               while (kill(pid, 0) == 0) {
                   std::this_thread::sleep_for(std::chrono::seconds(1));
               }
               detach_func();
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

Complete Tool Example
---------------------

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
       uint32_t duration = (argc > 2) ? std::stoi(argv[2]) : 0;

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
