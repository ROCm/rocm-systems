/*
 * Copyright © 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "libhsakmt.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdio.h>
#include "hsakmt/linux/kfd_ioctl.h"
#include "fmm.h"
#include "hsakmt/hsakmtmodel.h"

static HSAuint64 *events_page = NULL;

void hsakmt_clear_events_page(void)
{
	events_page = NULL;
}

static bool IsSystemEventType(HSA_EVENTTYPE type)
{
	// Debug events behave as signal events.
	return (type != HSA_EVENTTYPE_SIGNAL && type != HSA_EVENTTYPE_DEBUG_EVENT);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtCreateEvent(HsaEventDescriptor *EventDesc,
					  bool ManualReset, bool IsSignaled,
					  HsaEvent **Event)
{
	unsigned int event_limit = KFD_SIGNAL_EVENT_LIMIT;

	CHECK_KFD_OPEN();

	if (EventDesc->EventType >= HSA_EVENTTYPE_MAXID)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	HsaEvent *e = malloc(sizeof(HsaEvent));

	if (!e)
		return HSAKMT_STATUS_ERROR;

	memset(e, 0, sizeof(*e));

	struct kfd_ioctl_create_event_args args = {0};

	args.event_type = EventDesc->EventType;
	args.node_id = EventDesc->NodeId;
	args.auto_reset = !ManualReset;

	/* dGPU code */
	pthread_mutex_lock(&hsakmt_mutex);

	if (hsakmt_is_dgpu && !events_page) {
		events_page = hsakmt_allocate_exec_aligned_memory_gpu(
			KFD_SIGNAL_EVENT_LIMIT * 8, PAGE_SIZE, 0, 0, true, false, true);
		if (!events_page) {
			free(e);
			pthread_mutex_unlock(&hsakmt_mutex);
			return HSAKMT_STATUS_ERROR;
		}
		if (hsakmt_use_model)
			model_set_event_page(events_page, KFD_SIGNAL_EVENT_LIMIT);
		else
			hsakmt_fmm_get_handle(events_page, (uint64_t *)&args.event_page_offset);
	}

	if (hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_CREATE_EVENT, &args) != 0) {
		free(e);
		*Event = NULL;
		pthread_mutex_unlock(&hsakmt_mutex);
		return HSAKMT_STATUS_ERROR;
	}

	e->EventId = args.event_id;

	if (!events_page && args.event_page_offset > 0) {
		events_page = mmap(NULL, event_limit * 8, PROT_WRITE | PROT_READ,
				MAP_SHARED, hsakmt_kfd_fd, args.event_page_offset);
		if (events_page == MAP_FAILED) {
			/* old kernels only support 256 events */
			event_limit = 256;
			events_page = mmap(NULL, PAGE_SIZE, PROT_WRITE | PROT_READ,
					   MAP_SHARED, hsakmt_kfd_fd, args.event_page_offset);
		}
		if (events_page == MAP_FAILED) {
			events_page = NULL;
			pthread_mutex_unlock(&hsakmt_mutex);
			hsaKmtDestroyEvent(e);
			return HSAKMT_STATUS_ERROR;
		}
	}

	if (args.event_page_offset > 0 && args.event_slot_index < event_limit)
		e->EventData.HWData2 = (HSAuint64)&events_page[args.event_slot_index];

        pthread_mutex_unlock(&hsakmt_mutex);

        e->EventData.EventType = EventDesc->EventType;
        e->EventData.HWData1 = args.event_id;

	e->EventData.HWData3 = args.event_trigger_data;
	e->EventData.EventData.SyncVar.SyncVar.UserData =
		EventDesc->SyncVar.SyncVar.UserData;
	e->EventData.EventData.SyncVar.SyncVarSize =
		EventDesc->SyncVar.SyncVarSize;

	if (IsSignaled && !IsSystemEventType(e->EventData.EventType)) {
		struct kfd_ioctl_set_event_args set_args = {0};

		set_args.event_id = args.event_id;

                if (hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_SET_EVENT,
                                 &set_args) != 0) {
                  hsaKmtDestroyEvent(e);
                  return HSAKMT_STATUS_ERROR;
                }
        }

        *Event = e;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDestroyEvent(HsaEvent *Event)
{
	CHECK_KFD_OPEN();

	if (!Event)
		return HSAKMT_STATUS_INVALID_HANDLE;

	struct kfd_ioctl_destroy_event_args args = {0};

	args.event_id = Event->EventId;

	if (hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_DESTROY_EVENT, &args) != 0)
		return HSAKMT_STATUS_ERROR;

	free(Event);
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSetEvent(HsaEvent *Event)
{
	CHECK_KFD_OPEN();

	if (!Event)
		return HSAKMT_STATUS_INVALID_HANDLE;

	/* Although the spec is doesn't say, don't allow system-defined events
	 * to be signaled.
	 */
	if (IsSystemEventType(Event->EventData.EventType))
		return HSAKMT_STATUS_ERROR;

	struct kfd_ioctl_set_event_args args = {0};

	args.event_id = Event->EventId;

	if (hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_SET_EVENT, &args) == -1)
		return HSAKMT_STATUS_ERROR;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtResetEvent(HsaEvent *Event)
{
	CHECK_KFD_OPEN();

	if (!Event)
		return HSAKMT_STATUS_INVALID_HANDLE;

	/* Although the spec is doesn't say, don't allow system-defined events
	 * to be signaled.
	 */
	if (IsSystemEventType(Event->EventData.EventType))
		return HSAKMT_STATUS_ERROR;

	struct kfd_ioctl_reset_event_args args = {0};

	args.event_id = Event->EventId;

	if (hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_RESET_EVENT, &args) == -1)
		return HSAKMT_STATUS_ERROR;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtQueryEventState(HsaEvent *Event)
{
	CHECK_KFD_OPEN();

	if (!Event)
		return HSAKMT_STATUS_INVALID_HANDLE;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtWaitOnEvent(HsaEvent *Event,
		HSAuint32 Milliseconds)
{
	return hsaKmtWaitOnEvent_Ext(Event, Milliseconds, NULL);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtWaitOnEvent_Ext(HsaEvent *Event,
		HSAuint32 Milliseconds, uint64_t *event_age)
{
	if (!Event)
		return HSAKMT_STATUS_INVALID_HANDLE;

	return hsaKmtWaitOnMultipleEvents_Ext(&Event, 1, true, Milliseconds, event_age);
}

static HSAKMT_STATUS get_mem_info_svm_api(uint64_t address, uint32_t gpu_id)
{
	struct kfd_ioctl_svm_args *args;
        uint32_t node_id = 0;
        HSAuint32 s_attr;
        HSAuint32 i;
	HSA_SVM_ATTRIBUTE attrs[] = {
					{HSA_SVM_ATTR_PREFERRED_LOC, 0},
					{HSA_SVM_ATTR_PREFETCH_LOC, 0},
					{HSA_SVM_ATTR_ACCESS, gpu_id},
					{HSA_SVM_ATTR_SET_FLAGS, 0},
				    };

	CHECK_KFD_OPEN();
	CHECK_KFD_MINOR_VERSION(5);

	s_attr = sizeof(attrs);
	args = alloca(sizeof(*args) + s_attr);
	args->start_addr = address;
	args->size = PAGE_SIZE;
	args->op = KFD_IOCTL_SVM_OP_GET_ATTR;
	args->nattr = s_attr / sizeof(*attrs);
	memcpy(args->attrs, attrs, s_attr);
	if (hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_SVM + (s_attr << _IOC_SIZESHIFT), args)) {
		pr_debug("op get range attrs failed %s\n", strerror(errno));
		return HSAKMT_STATUS_ERROR;
	}

	pr_err("GPU address 0x%lx, is Unified memory\n", address);
	for (i = 0; i < args->nattr; i++) {
		if (args->attrs[i].value == KFD_IOCTL_SVM_LOCATION_SYSMEM ||
		    args->attrs[i].value == KFD_IOCTL_SVM_LOCATION_UNDEFINED)
			node_id = args->attrs[i].value;
		else
			hsakmt_gpuid_to_nodeid(args->attrs[i].value, &node_id);
		switch (args->attrs[i].type) {
		case KFD_IOCTL_SVM_ATTR_PREFERRED_LOC:
			pr_err("Preferred location for address 0x%lx is Node id %d\n",
				address, node_id);
			break;
		case KFD_IOCTL_SVM_ATTR_PREFETCH_LOC:
			pr_err("Prefetch location for address 0x%lx is Node id %d\n",
				address, node_id);
			break;
		case KFD_IOCTL_SVM_ATTR_ACCESS:
			pr_err("Node id %d has access to address 0x%lx\n",
				node_id, address);
			break;
		case KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE:
			pr_err("Node id %d has access in place to address 0x%lx\n",
				node_id, address);
			break;
		case KFD_IOCTL_SVM_ATTR_NO_ACCESS:
			pr_err("Node id %d has no access to address 0x%lx\n",
				node_id, address);
			break;
		case KFD_IOCTL_SVM_ATTR_SET_FLAGS:
			if (args->attrs[i].value & KFD_IOCTL_SVM_FLAG_COHERENT)
				pr_err("Fine grained coherency between devices\n");
			if (args->attrs[i].value & KFD_IOCTL_SVM_FLAG_GPU_RO)
				pr_err("Read only\n");
			if (args->attrs[i].value & KFD_IOCTL_SVM_FLAG_GPU_EXEC)
				pr_err("GPU exec allowed\n");
			if (args->attrs[i].value & KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED)
				 pr_err("GPU always mapped\n");
			if (args->attrs[i].value & KFD_IOCTL_SVM_FLAG_EXT_COHERENT)
				 pr_err("Extended-scope fine grained coherency between devices\n");
			break;
		default:
			pr_debug("get invalid attr type 0x%x\n", args->attrs[i].type);
			return HSAKMT_STATUS_ERROR;
		}
	}

	return HSAKMT_STATUS_SUCCESS;
}
//Analysis memory exception data, print debug messages
static void analysis_memory_exception(struct kfd_hsa_memory_exception_data *
						memory_exception_data)
{
	HSAKMT_STATUS ret;
	HsaPointerInfo info;
	const uint64_t addr = memory_exception_data->va;
	uint32_t node_id = 0;
	unsigned int i;

	hsakmt_gpuid_to_nodeid(memory_exception_data->gpu_id, &node_id);
	pr_err("Memory exception on virtual address 0x%lx, ", addr);
	pr_err("node id %d : ", node_id);
	if (memory_exception_data->failure.NotPresent)
		pr_err("Page not present\n");
	else if (memory_exception_data->failure.ReadOnly)
		pr_err("Writing to readonly page\n");
	else if (memory_exception_data->failure.NoExecute)
		pr_err("Execute to none-executable page\n");

	ret = hsakmt_fmm_get_mem_info((const void *)addr, &info);
	if (ret != HSAKMT_STATUS_SUCCESS) {
		ret = get_mem_info_svm_api(addr, memory_exception_data->gpu_id);
		if (ret != HSAKMT_STATUS_SUCCESS)
			pr_err("Address does not belong to a known buffer\n");
		return;
	}

	pr_err("GPU address 0x%lx, node id %d, size in byte 0x%lx\n",
			info.GPUAddress, info.Node, info.SizeInBytes);
	switch (info.Type) {
	case HSA_POINTER_REGISTERED_SHARED:
		pr_err("Memory is registered shared buffer (IPC)\n");
		break;
	case HSA_POINTER_REGISTERED_GRAPHICS:
		pr_err("Memory is registered graphics buffer\n");
		break;
	case HSA_POINTER_REGISTERED_USER:
		pr_err("Memory is registered user pointer\n");
		pr_err("CPU address of the memory is %p\n", info.CPUAddress);
		break;
	case HSA_POINTER_ALLOCATED:
		pr_err("Memory is allocated using hsaKmtAllocMemory\n");
		pr_err("CPU address of the memory is %p\n", info.CPUAddress);
		break;
	case HSA_POINTER_RESERVED_ADDR:
		pr_err("Memory is allocated by OnlyAddress mode\n");
		break;
	default:
		pr_err("Invalid memory type %d\n", info.Type);
		break;
	}

	if (info.RegisteredNodes) {
		pr_err("Memory is registered to node id: ");
		for (i = 0; i < info.NRegisteredNodes; i++)
			pr_err("%d ", info.RegisteredNodes[i]);
		pr_err("\n");
	}
	if (info.MappedNodes) {
		pr_err("Memory is mapped to node id: ");
		for (i = 0; i < info.NMappedNodes; i++)
			pr_err("%d ", info.MappedNodes[i]);
		pr_err("\n");
	}
}

HSAKMT_STATUS HSAKMTAPI hsaKmtWaitOnMultipleEvents(HsaEvent *Events[],
						   HSAuint32 NumEvents,
						   bool WaitOnAll,
						   HSAuint32 Milliseconds)
{
	return hsaKmtWaitOnMultipleEvents_Ext(Events, NumEvents, WaitOnAll, Milliseconds, NULL);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtWaitOnMultipleEvents_Ext(HsaEvent *Events[],
						   HSAuint32 NumEvents,
						   bool WaitOnAll,
						   HSAuint32 Milliseconds,
						   uint64_t *event_age)
{
        HSAKMT_STATUS result;
        CHECK_KFD_OPEN();

        if (!Events)
		return HSAKMT_STATUS_INVALID_HANDLE;

        struct kfd_event_data *event_data =
        calloc(NumEvents, sizeof(struct kfd_event_data));
        if (!event_data) {
		return HSAKMT_STATUS_NO_MEMORY;
	}
        for (HSAuint32 i = 0; i < NumEvents; i++) {
		event_data[i].event_id = Events[i]->EventId;
		event_data[i].kfd_event_data_ext = (uint64_t)(uintptr_t)NULL;
		if (event_age && Events[i]->EventData.EventType == HSA_EVENTTYPE_SIGNAL)
			event_data[i].signal_event_data.last_event_age = event_age[i];
	}

        struct kfd_ioctl_wait_events_args args = {0};

	args.wait_for_all = WaitOnAll;
	args.timeout = Milliseconds;
	args.num_events = NumEvents;
	args.events_ptr = (uint64_t)(uintptr_t)event_data;

	if (hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_WAIT_EVENTS, &args) == -1)
		result = HSAKMT_STATUS_ERROR;
	else if (args.wait_result == KFD_IOC_WAIT_RESULT_TIMEOUT)
		result = HSAKMT_STATUS_WAIT_TIMEOUT;
	else {
		result = HSAKMT_STATUS_SUCCESS;
		for (HSAuint32 i = 0; i < NumEvents; i++) {
			if (Events[i]->EventData.EventType == HSA_EVENTTYPE_MEMORY &&
			    event_data[i].memory_exception_data.gpu_id) {
				Events[i]->EventData.EventData.MemoryAccessFault.VirtualAddress = event_data[i].memory_exception_data.va;
				result = hsakmt_gpuid_to_nodeid(event_data[i].memory_exception_data.gpu_id, &Events[i]->EventData.EventData.MemoryAccessFault.NodeId);
				if (result != HSAKMT_STATUS_SUCCESS)
					goto out;
				Events[i]->EventData.EventData.MemoryAccessFault.Failure.NotPresent = event_data[i].memory_exception_data.failure.NotPresent;
				Events[i]->EventData.EventData.MemoryAccessFault.Failure.ReadOnly = event_data[i].memory_exception_data.failure.ReadOnly;
				Events[i]->EventData.EventData.MemoryAccessFault.Failure.NoExecute = event_data[i].memory_exception_data.failure.NoExecute;
				Events[i]->EventData.EventData.MemoryAccessFault.Failure.Imprecise = event_data[i].memory_exception_data.failure.imprecise;
				Events[i]->EventData.EventData.MemoryAccessFault.Failure.ErrorType = event_data[i].memory_exception_data.ErrorType;
				Events[i]->EventData.EventData.MemoryAccessFault.Failure.ECC =
						((event_data[i].memory_exception_data.ErrorType == 1) || (event_data[i].memory_exception_data.ErrorType == 2)) ? 1 : 0;
				Events[i]->EventData.EventData.MemoryAccessFault.Flags = HSA_EVENTID_MEMORY_FATAL_PROCESS;
				analysis_memory_exception(&event_data[i].memory_exception_data);
			} else if (Events[i]->EventData.EventType == HSA_EVENTTYPE_HW_EXCEPTION &&
				event_data[i].hw_exception_data.gpu_id) {

				result = hsakmt_gpuid_to_nodeid(event_data[i].hw_exception_data.gpu_id, &Events[i]->EventData.EventData.HwException.NodeId);
				if (result != HSAKMT_STATUS_SUCCESS)
					goto out;

				Events[i]->EventData.EventData.HwException.ResetType = event_data[i].hw_exception_data.reset_type;
				Events[i]->EventData.EventData.HwException.ResetCause = event_data[i].hw_exception_data.reset_cause;
				Events[i]->EventData.EventData.HwException.MemoryLost = event_data[i].hw_exception_data.memory_lost;
			}
		}
	}
out:

	for (HSAuint32 i = 0; i < NumEvents; i++) {
		if (event_age && Events[i]->EventData.EventType == HSA_EVENTTYPE_SIGNAL)
			event_age[i] = event_data[i].signal_event_data.last_event_age;
	}

	free(event_data);

	return result;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtOpenSMI(HSAuint32 NodeId, int *fd)
{
	struct kfd_ioctl_smi_events_args args;
	HSAKMT_STATUS result;
	uint32_t gpuid;

	CHECK_KFD_OPEN();

	pr_debug("[%s] node %d\n", __func__, NodeId);

	result = hsakmt_validate_nodeid(NodeId, &gpuid);
	if (result != HSAKMT_STATUS_SUCCESS) {
		pr_err("[%s] invalid node ID: %d\n", __func__, NodeId);
		return result;
	}

	args.gpuid = gpuid;
	result = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_SMI_EVENTS, &args);
	if (result) {
		pr_debug("open SMI event fd failed %s\n", strerror(errno));
		return HSAKMT_STATUS_ERROR;
	}

	*fd = args.anon_fd;
	return HSAKMT_STATUS_SUCCESS;
}
