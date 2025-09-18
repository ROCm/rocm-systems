.. meta::
  :description: Technical guide for implementing ROCprofiler-SDK process attachment
  :keywords: ROCprofiler-SDK, process attachment, ptrace, dynamic profiling, tool development

.. _process_attachment_implementation:

********************************************************************************
Implementing Process Attachment Tools
********************************************************************************

Overview
========

This document provides the technical details needed to implement a process attachment tool similar to ``rocprofv3 --attach``. Process attachment allows profiling tools to dynamically attach to running GPU applications without requiring application restart.

The implementation uses specific exported C functions and involves low-level process manipulation using ptrace, environment variable injection, library loading, and coordination with the ROCprofiler-SDK registration system.

Exported C Functions for Attachment
===================================

The attachment functionality provides the following exported C functions that tools can use:

ROCprofiler-Attach Functions
-----------------------------

These functions are exported from the ``rocprofiler-attach`` binary:

.. code-block:: cpp

   extern "C" {
       // Start attachment to a target process
       void attach(uint32_t pid) ROCPROFILER_EXPORT;

       // Detach from target process and cleanup
       void detach() ROCPROFILER_EXPORT;
   }

**Function Details:**

- **``attach(uint32_t pid)``**: Main entry point for starting attachment to a process
  - Takes the target process ID as parameter
  - Initiates ptrace-based attachment sequence
  - Spawns background thread for ptrace operations

- **``detach()``**: Entry point for detaching from the target process
  - Cleans up attachment resources and terminates profiling
  - Joins ptrace thread and releases resources

ROCprofiler-Register Functions
------------------------------

These functions are exported from the ``librocprofiler-register.so`` library and are called via ptrace:

.. code-block:: cpp

   extern "C" {
       // Activate profiling in target process (called via ptrace)
       rocprofiler_register_error_code_t
       rocprofiler_register_attach(const char* environment_buffer, const char* tool_lib_path)
           ROCPROFILER_REGISTER_PUBLIC_API;

       // Deactivate profiling in target process (called via ptrace)
       rocprofiler_register_error_code_t
       rocprofiler_register_detach()
           ROCPROFILER_REGISTER_PUBLIC_API;

       // Reattach to previously attached process (experimental)
       rocprofiler_register_error_code_t
       rocprofiler_register_invoke_reattach()
           ROCPROFILER_REGISTER_PUBLIC_API;

       // Client callback functions for reattachment support
       void rocprofiler_call_client_reattach(void)
           ROCPROFILER_REGISTER_PUBLIC_API;
       void rocprofiler_call_client_detach(void)
           ROCPROFILER_REGISTER_PUBLIC_API;
   }

**Function Details:**

- **``rocprofiler_register_attach(const char* environment_buffer, const char* tool_lib_path)``**:
  - Called via ptrace from the attachment system
  - Receives serialized environment variables for profiling configuration
  - Receives the tool library path to load (defaults to "librocprofiler-sdk-tool.so" if NULL)
  - Loads the specified tool library and activates profiling services
  - Returns ``rocprofiler_register_error_code_t`` status

- **``rocprofiler_register_detach()``**:
  - Called via ptrace to stop profiling in the target process
  - Calls the tool's detach function and cleans up resources
  - Returns ``rocprofiler_register_error_code_t`` status

- **``rocprofiler_register_invoke_reattach()``**: (EXPERIMENTAL)
  - Called to reattach profiling to a previously attached process
  - Invokes client reattach callbacks without full re-initialization
  - Used for resuming profiling after temporary detachment
  - Returns ``rocprofiler_register_error_code_t`` status

- **``rocprofiler_call_client_reattach()`` and ``rocprofiler_call_client_detach()``**:
  - C wrapper functions for client tool reattachment callbacks
  - Automatically resolved and called by the registration system
  - Enable tools to handle dynamic attach/detach cycles

Function Call Sequence
======================

Initial Attachment Sequence
---------------------------

The initial attachment process follows this sequence:

.. code-block:: text

   Tool Implementation
        |
        v
   attach(pid) ← Your tool calls this
        |
        v
   Ptrace attachment & environment setup
        |
        v
   rocprofiler_register_attach(env_buffer) ← Called via ptrace in target
        |
        v
   Profiling active in target process
        |
        v
   [Profiling data collection...]
        |
        v
   rocprofiler_register_detach() ← Called via ptrace in target
        |
        v
   detach() ← Your tool calls this
        |
        v
   Cleanup complete

Reattachment Sequence (Experimental)
------------------------------------

For reattachment to a previously attached process:

.. code-block:: text

   Tool Implementation
        |
        v
   attach(pid) ← Your tool calls this again
        |
        v
   Ptrace attachment & environment setup
        |
        v
   rocprofiler_register_attach(env_buffer) ← Detects previous attachment
        |
        v
   rocprofiler_register_invoke_reattach() ← Calls client reattach callbacks
        |
        v
   Profiling resumed in target process
        |
        v
   [Continued profiling data collection...]
        |
        v
   rocprofiler_register_detach() ← Called via ptrace in target
        |
        v
   detach() ← Your tool calls this
        |
        v
   Cleanup complete

Using the Attachment Functions
==============================

Here's how to use these functions in your own attachment tool:

Basic Attachment Tool Implementation
-----------------------------------

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
           attach_lib_handle = dlopen("librocprofiler-attach.so", RTLD_NOW);
           if (!attach_lib_handle) {
               std::cerr << "Failed to load rocprofiler-attach: " << dlerror() << std::endl;
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
--------------------

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

       // Set up profiling environment variables before attachment
       setenv("ROCP_TOOL_ATTACH", "1", 1);

       // Note: The attachment system now uses the hardcoded default tool library path
       // "librocprofiler-sdk-tool.so" and no longer uses environment variables for tool selection

       setenv("ROCPROF_HIP_API_TRACE", "1", 1);
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

Experimental Reattachment API
=============================

ROCprofiler-SDK now provides experimental support for reattachment, allowing tools to handle dynamic attach/detach cycles more efficiently.

Tool Configuration for Reattachment
-----------------------------------

Tools that support reattachment should implement the experimental configuration structure:

.. code-block:: cpp

   #include <rocprofiler-sdk/registration.h>

   // Experimental reattachment callbacks
   void tool_reattach(void* tool_data) {
       // Reinitialize contexts and resume profiling
       // This is called when reattaching to a previously profiled process
   }

   void tool_detach(void* tool_data) {
       // Suspend profiling operations temporarily
       // This is called during detachment, but contexts may be preserved
   }

   extern "C" rocprofiler_tool_configure_result_experimental_t*
   rocprofiler_configure_experimental(uint32_t                 version,
                                      const char*              runtime_version,
                                      uint32_t                 prio,
                                      rocprofiler_client_id_t* client_id)
   {
       static auto cfg = rocprofiler_tool_configure_result_experimental_t {
           .size = sizeof(rocprofiler_tool_configure_result_experimental_t),
           .initialize = &tool_init,
           .finalize = &tool_fini,
           .tool_data = nullptr,
           .tool_reattach = &tool_reattach,  // Experimental reattachment support
           .tool_detach = &tool_detach       // Experimental detachment support
       };

       return &cfg;
   }

Client Callback Functions
-------------------------

The registration system automatically provides C wrapper functions:

.. code-block:: cpp

   // These are automatically generated and called by rocprofiler-register
   extern "C" void rocprofiler_call_client_reattach(void) {
       // Calls the tool's reattach callback with stored tool_data
   }

   extern "C" void rocprofiler_call_client_detach(void) {
       // Calls the tool's detach callback with stored tool_data
   }

Reattachment Environment Variables
---------------------------------

When using reattachment, set this additional environment variable:

.. code-block:: cpp

   // Indicates that the tool was loaded via attachment (not LD_PRELOAD)
   setenv("ROCPROFILER_REGISTER_TOOL_ATTACHED", "1", 1);

This helps the registration system differentiate between initial attachment and reattachment cycles.

Environment Variable Configuration
=================================

Before calling the attachment functions, set up environment variables that will be injected into the target process:

Required Variables
-----------------

.. code-block:: cpp

   // Essential for attachment functionality
   setenv("ROCP_TOOL_ATTACH", "1", 1);

Tool Library Configuration
--------------------------

The attachment system now uses a hardcoded default tool library path:

.. code-block:: cpp

   // The attachment system automatically uses "librocprofiler-sdk-tool.so"
   // No environment variable configuration is needed or supported

Tracing Options
--------------

.. code-block:: cpp

   // Enable different types of tracing
   setenv("ROCPROF_HIP_API_TRACE", "1", 1);           // HIP API calls
   setenv("ROCPROF_HSA_API_TRACE", "1", 1);           // HSA API calls
   setenv("ROCPROF_KERNEL_TRACE", "1", 1);            // Kernel dispatches
   setenv("ROCPROF_MEMORY_COPY_TRACE", "1", 1);       // Memory operations
   setenv("ROCPROF_MEMORY_ALLOCATION_TRACE", "1", 1); // Memory allocations
   setenv("ROCPROF_SCRATCH_MEMORY_TRACE", "1", 1);    // Scratch memory
   setenv("ROCPROF_MARKER_TRACE", "1", 1);            // ROCTx markers

Output Configuration
-------------------

.. code-block:: cpp

   // Control output location and format
   setenv("ROCPROF_OUTPUT_PATH", "/path/to/output", 1);
   setenv("ROCPROF_OUTPUT_FILE_NAME", "profile_name", 1);
   setenv("ROCPROF_OUTPUT_FORMAT", "csv", 1);  // or "json", "pftrace", etc.

Build Configuration
==================

To build a tool using the attachment functions:

CMakeLists.txt
-------------

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.16)
   project(my_rocprofiler_attach_tool)

   set(CMAKE_CXX_STANDARD 17)

   # Find ROCprofiler SDK (for headers and linking)
   find_package(rocprofiler-sdk REQUIRED)

   add_executable(my_attach_tool
       main.cpp
       attachment_tool.cpp
   )

   # Link with required libraries
   target_link_libraries(my_attach_tool
       rocprofiler-sdk::rocprofiler-sdk
       dl  # for dlopen/dlsym operations
   )

   # Set capabilities for ptrace operations
   add_custom_command(TARGET my_attach_tool POST_BUILD
       COMMAND sudo setcap cap_sys_ptrace+ep $<TARGET_FILE:my_attach_tool>
       COMMENT "Setting ptrace capability"
   )

Error Handling
=============

When using the attachment functions, handle these common error conditions:

.. code-block:: cpp

   class AttachmentErrorHandler {
   public:
       static bool validate_target_process(pid_t pid) {
           // Check if process exists
           if (kill(pid, 0) != 0) {
               std::cerr << "Process " << pid << " not found or not accessible" << std::endl;
               return false;
           }

           // Check if it's a GPU application
           std::string maps_path = "/proc/" + std::to_string(pid) + "/maps";
           std::ifstream maps(maps_path);
           std::string line;

           bool has_gpu_libs = false;
           while (std::getline(maps, line)) {
               if (line.find("libamdhip64.so") != std::string::npos ||
                   line.find("libhsa-runtime64.so") != std::string::npos) {
                   has_gpu_libs = true;
                   break;
               }
           }

           if (!has_gpu_libs) {
               std::cerr << "Process " << pid << " does not appear to use GPU APIs" << std::endl;
               return false;
           }

           return true;
       }

       static void handle_attachment_errors() {
           // Check for common permission issues
           if (geteuid() != 0) {
               std::cerr << "Warning: Not running as root. Ensure CAP_SYS_PTRACE capability is set." << std::endl;
           }

           // Check if rocprofiler libraries are available
           if (getenv("LD_LIBRARY_PATH") == nullptr ||
               std::string(getenv("LD_LIBRARY_PATH")).find("/opt/rocm/lib") == std::string::npos) {
               std::cerr << "Warning: /opt/rocm/lib may not be in LD_LIBRARY_PATH" << std::endl;
           }
       }
   };

Architecture Overview
=====================

Process attachment consists of several cooperating components:

.. code-block:: text

   Attachment Tool (your implementation)
        |
        v
   1. Process Discovery & Validation
        |
        v
   2. Ptrace Attachment & Control
        |
        v
   3. Environment Variable Injection
        |
        v
   4. Library Loading (rocprofiler-register)
        |
        v
   5. Profiling Service Activation
        |
        v
   6. Data Collection & Management
        |
        v
   7. Detachment & Cleanup

Theoretical Implementation Details
=================================

Core Implementation Components
=============================

1. Process Discovery and Validation
-----------------------------------

**Target Process Requirements:**

.. code-block:: cpp

   #include <sys/types.h>
   #include <signal.h>
   #include <unistd.h>

   bool validate_target_process(pid_t pid) {
       // Check if process exists and is accessible
       if (kill(pid, 0) != 0) {
           return false;  // Process doesn't exist or no permission
       }

       // Verify it's a GPU application by checking loaded libraries
       std::string maps_path = "/proc/" + std::to_string(pid) + "/maps";
       std::ifstream maps(maps_path);
       std::string line;

       bool has_hip = false, has_hsa = false;
       while (std::getline(maps, line)) {
           if (line.find("libamdhip64.so") != std::string::npos) has_hip = true;
           if (line.find("libhsa-runtime64.so") != std::string::npos) has_hsa = true;
       }

       return has_hip || has_hsa;  // Must use HIP or HSA
   }

2. Ptrace-Based Process Control
------------------------------

**Core Ptrace Operations:**

.. code-block:: cpp

   #include <sys/ptrace.h>
   #include <sys/wait.h>
   #include <sys/user.h>

   class ProcessAttachment {
   private:
       pid_t target_pid;
       bool attached = false;

   public:
       bool attach(pid_t pid) {
           target_pid = pid;

           // Attach to the target process
           if (ptrace(PTRACE_ATTACH, target_pid, nullptr, nullptr) == -1) {
               perror("ptrace PTRACE_ATTACH failed");
               return false;
           }

           // Wait for the process to stop
           int status;
           if (waitpid(target_pid, &status, 0) == -1) {
               perror("waitpid failed");
               detach();
               return false;
           }

           if (!WIFSTOPPED(status)) {
               fprintf(stderr, "Process did not stop after attach\n");
               detach();
               return false;
           }

           attached = true;
           return true;
       }

       bool detach() {
           if (!attached) return true;

           // Detach and allow process to continue
           if (ptrace(PTRACE_DETACH, target_pid, nullptr, nullptr) == -1) {
               perror("ptrace PTRACE_DETACH failed");
               return false;
           }

           attached = false;
           return true;
       }
   };

3. Environment Variable Injection
---------------------------------

**Environment Variable Management:**

.. code-block:: cpp

   #include <fstream>
   #include <vector>

   class EnvironmentInjector {
   public:
       struct EnvironmentVar {
           std::string name;
           std::string value;
       };

       // Prepare environment variables for profiling
       std::vector<EnvironmentVar> prepare_profiling_env(
           const std::vector<std::string>& trace_options,
           const std::string& output_path,
           const std::string& output_file) {

           std::vector<EnvironmentVar> env_vars;

           // Essential attachment variable
           env_vars.push_back({"ROCP_TOOL_ATTACH", "1"});

           // Configure tracing based on options
           for (const auto& option : trace_options) {
               if (option == "hip-trace") {
                   env_vars.push_back({"ROCPROF_HIP_API_TRACE", "1"});
               }
               if (option == "kernel-trace") {
                   env_vars.push_back({"ROCPROF_KERNEL_TRACE", "1"});
               }
               if (option == "hsa-trace") {
                   env_vars.push_back({"ROCPROF_HSA_API_TRACE", "1"});
               }
               if (option == "memory-copy-trace") {
                   env_vars.push_back({"ROCPROF_MEMORY_COPY_TRACE", "1"});
               }
           }

           // Output configuration
           env_vars.push_back({"ROCPROF_OUTPUT_PATH", output_path});
           env_vars.push_back({"ROCPROF_OUTPUT_FILE_NAME", output_file});

           return env_vars;
       }

       // Serialize environment for injection
       std::vector<uint8_t> serialize_environment(const std::vector<EnvironmentVar>& vars) {
           std::vector<uint8_t> buffer(4);  // Start with count
           uint32_t count = vars.size();

           // Store count in first 4 bytes
           buffer[0] = count & 0xFF;
           buffer[1] = (count >> 8) & 0xFF;
           buffer[2] = (count >> 16) & 0xFF;
           buffer[3] = (count >> 24) & 0xFF;

           // Add each variable as null-terminated name and value
           for (const auto& var : vars) {
               // Add variable name
               for (char c : var.name) {
                   buffer.push_back(c);
               }
               buffer.push_back(0);  // Null terminate name

               // Add variable value
               for (char c : var.value) {
                   buffer.push_back(c);
               }
               buffer.push_back(0);  // Null terminate value
           }

           return buffer;
       }
   };

4. Memory Manipulation and Library Loading
------------------------------------------

**Remote Memory Operations:**

.. code-block:: cpp

   #include <sys/mman.h>

   class RemoteMemoryManager {
   private:
       pid_t target_pid;

   public:
       RemoteMemoryManager(pid_t pid) : target_pid(pid) {}

       // Allocate memory in remote process
       void* remote_mmap(size_t length, int prot, int flags) {
           // Find a suitable location for injection
           struct user_regs_struct regs;
           if (ptrace(PTRACE_GETREGS, target_pid, nullptr, &regs) == -1) {
               return nullptr;
           }

           // Save original registers
           struct user_regs_struct orig_regs = regs;

           // Set up mmap syscall
           regs.rax = 9;  // __NR_mmap
           regs.rdi = 0;  // addr (let kernel choose)
           regs.rsi = length;
           regs.rdx = prot;
           regs.r10 = flags;
           regs.r8 = -1;  // fd
           regs.r9 = 0;   // offset

           if (ptrace(PTRACE_SETREGS, target_pid, nullptr, &regs) == -1) {
               return nullptr;
           }

           // Execute syscall
           if (ptrace(PTRACE_SYSCALL, target_pid, nullptr, nullptr) == -1) {
               return nullptr;
           }

           // Wait for syscall completion
           int status;
           waitpid(target_pid, &status, 0);

           // Get result
           if (ptrace(PTRACE_GETREGS, target_pid, nullptr, &regs) == -1) {
               return nullptr;
           }

           void* result = (void*)regs.rax;

           // Restore original registers
           ptrace(PTRACE_SETREGS, target_pid, nullptr, &orig_regs);

           return (result == (void*)-1) ? nullptr : result;
       }

       // Write data to remote process memory
       bool write_memory(void* addr, const void* data, size_t size) {
           const uint8_t* bytes = static_cast<const uint8_t*>(data);
           size_t written = 0;

           while (written < size) {
               long word = 0;
               size_t to_copy = std::min(sizeof(long), size - written);

               // For partial words, read existing content first
               if (to_copy < sizeof(long)) {
                   errno = 0;
                   word = ptrace(PTRACE_PEEKDATA, target_pid,
                                (uint8_t*)addr + written, nullptr);
                   if (errno != 0) return false;
               }

               // Copy new data into word
               memcpy(&word, bytes + written, to_copy);

               // Write word to remote process
               if (ptrace(PTRACE_POKEDATA, target_pid,
                         (uint8_t*)addr + written, word) == -1) {
                   return false;
               }

               written += to_copy;
           }

           return true;
       }
   };

5. Library Injection and Symbol Resolution
------------------------------------------

**Dynamic Library Loading:**

.. code-block:: cpp

   #include <dlfcn.h>
   #include <link.h>

   class LibraryInjector {
   private:
       pid_t target_pid;
       RemoteMemoryManager memory_manager;

   public:
       LibraryInjector(pid_t pid) : target_pid(pid), memory_manager(pid) {}

       // Inject rocprofiler-register library
       bool inject_register_library() {
           const char* lib_path = "/opt/rocm/lib/librocprofiler-register.so";

           // Find dlopen in target process
           void* dlopen_addr = find_function_address("dlopen");
           if (!dlopen_addr) {
               fprintf(stderr, "Could not find dlopen in target process\n");
               return false;
           }

           // Allocate memory for library path
           void* path_addr = memory_manager.remote_mmap(
               strlen(lib_path) + 1,
               PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS);

           if (!path_addr) return false;

           // Write library path to remote memory
           if (!memory_manager.write_memory(path_addr, lib_path, strlen(lib_path) + 1)) {
               return false;
           }

           // Call dlopen in target process
           return call_remote_function(dlopen_addr,
               {(uint64_t)path_addr, RTLD_NOW | RTLD_GLOBAL});
       }

       void* find_function_address(const char* function_name) {
           // Parse /proc/PID/maps to find loaded libraries
           std::string maps_path = "/proc/" + std::to_string(target_pid) + "/maps";
           std::ifstream maps(maps_path);
           std::string line;

           while (std::getline(maps, line)) {
               if (line.find("libc.so") != std::string::npos) {
                   // Extract base address of libc
                   size_t dash = line.find('-');
                   std::string base_addr_str = line.substr(0, dash);
                   void* base_addr = (void*)std::stoull(base_addr_str, nullptr, 16);

                   // Open libc and find function offset
                   void* handle = dlopen("libc.so.6", RTLD_LAZY);
                   if (handle) {
                       void* func_addr = dlsym(handle, function_name);
                       if (func_addr) {
                           // Calculate actual address in target process
                           return (uint8_t*)base_addr + ((uint8_t*)func_addr - (uint8_t*)dlsym(RTLD_DEFAULT, "main"));
                       }
                       dlclose(handle);
                   }
               }
           }
           return nullptr;
       }
   };

6. ROCprofiler-Register Communication Protocol
----------------------------------------------

**Attachment Protocol Implementation:**

.. code-block:: cpp

   extern "C" {
       // Function signatures from rocprofiler-register
       typedef void (*attach_func_t)(uint32_t pid);
       typedef void (*detach_func_t)();
   }

   class ROCprofilerAttachment {
   private:
       pid_t target_pid;
       void* register_handle = nullptr;
       attach_func_t attach_func = nullptr;
       detach_func_t detach_func = nullptr;

   public:
       bool initialize() {
           // Load rocprofiler-register library
           register_handle = dlopen("/opt/rocm/lib/librocprofiler-register.so", RTLD_NOW);
           if (!register_handle) {
               fprintf(stderr, "Failed to load rocprofiler-register: %s\n", dlerror());
               return false;
           }

           // Get attachment functions
           attach_func = (attach_func_t)dlsym(register_handle, "attach");
           detach_func = (detach_func_t)dlsym(register_handle, "detach");

           if (!attach_func || !detach_func) {
               fprintf(stderr, "Failed to find attachment functions\n");
               return false;
           }

           return true;
       }

       bool attach_to_process(pid_t pid, const std::vector<uint8_t>& env_buffer) {
           target_pid = pid;

           // Set up environment for rocprofiler-register
           // This involves injecting the environment buffer into the target process

           // Call the attach function
           attach_func(pid);

           return true;
       }

       void detach_from_process() {
           if (detach_func) {
               detach_func();
           }
       }
   };

Complete Attachment Tool Implementation
======================================

**Main Attachment Tool Structure:**

.. code-block:: cpp

   #include <iostream>
   #include <vector>
   #include <string>
   #include <chrono>
   #include <thread>

   class ROCprofilerAttachTool {
   private:
       ProcessAttachment process_control;
       EnvironmentInjector env_injector;
       LibraryInjector lib_injector;
       ROCprofilerAttachment rocprof_attachment;

   public:
       struct AttachmentConfig {
           pid_t target_pid;
           std::vector<std::string> trace_options;
           std::string output_path = "./rocprof-attachment-output";
           std::string output_filename = "attached_profile";
           uint32_t duration_msec = 0;  // 0 = until process ends
       };

       bool attach_and_profile(const AttachmentConfig& config) {
           // 1. Validate target process
           if (!validate_target_process(config.target_pid)) {
               std::cerr << "Invalid or inaccessible target process: " << config.target_pid << std::endl;
               return false;
           }

           // 2. Initialize rocprofiler attachment system
           if (!rocprof_attachment.initialize()) {
               std::cerr << "Failed to initialize rocprofiler attachment system" << std::endl;
               return false;
           }

           // 3. Attach to target process
           if (!process_control.attach(config.target_pid)) {
               std::cerr << "Failed to attach to process " << config.target_pid << std::endl;
               return false;
           }

           // 4. Prepare environment variables
           auto env_vars = env_injector.prepare_profiling_env(
               config.trace_options,
               config.output_path,
               config.output_filename);
           auto env_buffer = env_injector.serialize_environment(env_vars);

           // 5. Inject rocprofiler-register library
           LibraryInjector injector(config.target_pid);
           if (!injector.inject_register_library()) {
               std::cerr << "Failed to inject rocprofiler-register library" << std::endl;
               process_control.detach();
               return false;
           }

           // 6. Activate profiling
           if (!rocprof_attachment.attach_to_process(config.target_pid, env_buffer)) {
               std::cerr << "Failed to activate profiling" << std::endl;
               process_control.detach();
               return false;
           }

           // 7. Allow process to continue with profiling active
           if (!process_control.detach()) {
               std::cerr << "Warning: Failed to detach cleanly" << std::endl;
           }

           // 8. Wait for specified duration or until process ends
           if (config.duration_msec > 0) {
               std::cout << "Profiling for " << config.duration_msec << " milliseconds..." << std::endl;
               std::this_thread::sleep_for(std::chrono::milliseconds(config.duration_msec));

               // Re-attach to stop profiling
               rocprof_attachment.detach_from_process();
           } else {
               std::cout << "Profiling until process ends..." << std::endl;
               // Monitor process and wait for it to end
               while (kill(config.target_pid, 0) == 0) {
                   std::this_thread::sleep_for(std::chrono::seconds(1));
               }
           }

           std::cout << "Profiling completed. Output saved to: "
                     << config.output_path << "/" << config.output_filename << std::endl;
           return true;
       }
   };

   // Example usage
   int main(int argc, char* argv[]) {
       if (argc < 2) {
           std::cerr << "Usage: " << argv[0] << " <PID> [options]" << std::endl;
           return 1;
       }

       ROCprofilerAttachTool::AttachmentConfig config;
       config.target_pid = std::stoi(argv[1]);
       config.trace_options = {"hip-trace", "kernel-trace", "memory-copy-trace"};
       config.duration_msec = 5000;  // 5 seconds

       ROCprofilerAttachTool tool;
       if (!tool.attach_and_profile(config)) {
           std::cerr << "Attachment and profiling failed" << std::endl;
           return 1;
       }

       return 0;
   }

Required System Permissions and Setup
=====================================

**Permission Requirements:**

.. code-block:: bash

   # Your attachment tool will need:

   # 1. Ptrace permissions (may require root or capabilities)
   sudo setcap cap_sys_ptrace+ep your_attachment_tool

   # 2. Access to /proc filesystem
   # Usually available by default

   # 3. Ability to load shared libraries
   # Ensure ROCm libraries are in LD_LIBRARY_PATH
   export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

**Build Requirements:**

.. code-block:: cmake

   # CMakeLists.txt for your attachment tool
   cmake_minimum_required(VERSION 3.16)
   project(rocprofiler_attach_tool)

   set(CMAKE_CXX_STANDARD 17)

   find_package(rocprofiler-sdk REQUIRED)

   add_executable(rocprofiler_attach_tool
       main.cpp
       process_attachment.cpp
       environment_injection.cpp
       library_injection.cpp
   )

   target_link_libraries(rocprofiler_attach_tool
       rocprofiler-sdk::rocprofiler-sdk
       dl  # for dlopen/dlsym
   )

Error Handling and Debugging
============================

**Common Issues and Solutions:**

1. **Ptrace Permissions**: Use ``strace`` to debug ptrace failures
2. **Library Loading**: Check ``/proc/PID/maps`` to verify library injection
3. **Environment Variables**: Validate environment buffer format
4. **Process State**: Monitor target process status during attachment

**Debugging Techniques:**

.. code-block:: cpp

   // Enable debug logging
   setenv("ROCPROF_LOGGING_LEVEL", "trace", 1);

   // Monitor attachment progress
   bool debug_attachment(pid_t pid) {
       std::cout << "Target process memory maps:" << std::endl;
       std::string cmd = "cat /proc/" + std::to_string(pid) + "/maps";
       system(cmd.c_str());

       std::cout << "Target process environment:" << std::endl;
       cmd = "cat /proc/" + std::to_string(pid) + "/environ | tr '\\0' '\\n'";
       system(cmd.c_str());

       return true;
   }

This implementation guide provides the foundation needed to build a complete process attachment tool for ROCprofiler-SDK. The actual rocprofv3 implementation uses similar techniques with additional optimizations and error handling.
