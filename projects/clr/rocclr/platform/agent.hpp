/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef AGENT_HPP_
#define AGENT_HPP_

#include "top.hpp"
#include "thread/monitor.hpp"

#include "vdi_agent_amd.h"

namespace amd {

class Agent : public _vdi_agent {
 public:
  //! Pointer to the ICD dispatch table (cl_icd_dispatch[]); set by the opencl layer.
  //! Stored as void* so rocclr doesn't need to include <CL/cl_icd.h>.
  static void* icdDispatch_;

 private:
  //! Linked list of agent instances
  static Agent* list_;
  //! Agent API entry points
  static const vdi_agent entryPoints_;
  //! Capabilities supported by this Agent implementation
  static vdi_agent_capabilities potentialCapabilities_;
  //! Union of all agent's enabled capabilities
  static vdi_agent_capabilities enabledCapabilities_;
  //! Monitor to protect the global capabilities
  static Monitor capabilitiesLock_;

 public:
  //! Initialize the OpenCL agent
  static bool init();
  //! Teardown the agent.
  static void tearDown();
  //! Return the capabilities supported by this agent.
  static vdi_agent_capabilities potentialCapabilities() { return potentialCapabilities_; }

#define AGENT_FLAG(name)                                                                           \
  inline static bool shouldPost##name() { return enabledCapabilities_.canGenerate##name != 0; }

  AGENT_FLAG(ContextEvents);
  AGENT_FLAG(CommandQueueEvents);
  AGENT_FLAG(EventEvents);
  AGENT_FLAG(MemObjectEvents);
  AGENT_FLAG(SamplerEvents);
  AGENT_FLAG(ProgramEvents);
  AGENT_FLAG(KernelEvents);

#undef AGENT_FLAG

  //! Post a context creation event
  static void postContextCreate(void* context);
  //! Post a context destruction event
  static void postContextFree(void* context);

  //! Post a command queue creation event
  static void postCommandQueueCreate(void* queue);
  //! Post a command queue destruction event
  static void postCommandQueueFree(void* queue);

  //! Post an event creation event
  static void postEventCreate(void* event, amd::CommandType type);
  //! Post an event destruction event
  static void postEventFree(void* event);
  //! Post and event status change event.
  static void postEventStatusChanged(void* event, int32_t execution_status,
                                     int64_t epoch_timestamp);

  //! Post a memory object creation event
  static void postMemObjectCreate(void* memobj);
  //! Post a memory object destruction event
  static void postMemObjectFree(void* memobj);
  //! Post a memory transfer (acquired by device) event
  static void postMemObjectAcquired(void* memobj, void* device, int64_t elapsed_time);

  //! Post a sampler creation event
  static void postSamplerCreate(void* sampler);
  //! Post a sampler destruction event
  static void postSamplerFree(void* sampler);

  //! Post a program creation event
  static void postProgramCreate(void* program);
  //! Post a program destruction event
  static void postProgramFree(void* program);
  //! Post a program build event
  static void postProgramBuild(void* program);

  //! Post a kernel creation event
  static void postKernelCreate(void* kernel);
  //! Post a kernel destruction event
  static void postKernelFree(void* kernel);
  //! Post a kernel set argument event
  static void postKernelSetArg(void* kernel, int32_t arg_index, size_t size,
                               const void* value_ptr);

 private:
  Agent* next_;    //!< Next agent in the linked-list.
  void* library_;  //!< Handle to the loaded module.
  bool ready_;     //!< Is this instance ready?

  //! Callbacks vector.
  vdi_agent_callbacks callbacks_;
  //! Capabilities for this agent.
  vdi_agent_capabilities capabilities_;

#define AGENT_FLAG(name)                                                                           \
  inline bool canGenerate##name() { return capabilities_.canGenerate##name != 0; }

  AGENT_FLAG(ContextEvents);
  AGENT_FLAG(CommandQueueEvents);
  AGENT_FLAG(EventEvents);
  AGENT_FLAG(MemObjectEvents);
  AGENT_FLAG(SamplerEvents);
  AGENT_FLAG(ProgramEvents);
  AGENT_FLAG(KernelEvents);

#undef AGENT_FLAG

 public:
  //! Construct a new agent.
  Agent(const char* moduleName);
  //! Destroy the agent
  ~Agent();

  //! Return true if this instance is ready for use.
  bool isReady() const { return ready_; }

  //! Set the callback vector for this agent
  int32_t setCallbacks(const vdi_agent_callbacks* callbacks, size_t size);

  //! Return the current capabilities.
  int32_t getCapabilities(vdi_agent_capabilities* caps);
  //! Set the current capabilities.
  int32_t setCapabilities(const vdi_agent_capabilities* caps, bool install);

  //! Return the Agent instance from the given cl_agent
  inline static Agent* get(vdi_agent* agent) {
    return const_cast<Agent*>(static_cast<const Agent*>(agent));
  }
};

}  // namespace amd

#endif  // AGENT_HPP_
