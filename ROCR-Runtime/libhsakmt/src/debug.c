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
#include "hsakmt/linux/kfd_ioctl.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static bool *is_device_debugged;
static uint32_t runtime_capabilities_mask = 0;

HSAKMT_STATUS hsakmt_init_device_debugging_memory(unsigned int NumNodes)
{
	unsigned int i;

	is_device_debugged = malloc(NumNodes * sizeof(bool));
	if (!is_device_debugged)
		return HSAKMT_STATUS_NO_MEMORY;

	for (i = 0; i < NumNodes; i++)
		is_device_debugged[i] = false;

	return HSAKMT_STATUS_SUCCESS;
}

void hsakmt_destroy_device_debugging_memory(void)
{
	if (is_device_debugged) {
		free(is_device_debugged);
		is_device_debugged = NULL;
	}
}

bool hsakmt_debug_get_reg_status(uint32_t node_id)
{
	return is_device_debugged[node_id];
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDbgRegister(HSAuint32 NodeId)
{
	HSAKMT_STATUS result;
	uint32_t gpu_id;

	CHECK_KFD_OPEN();

	if (!is_device_debugged)
		return HSAKMT_STATUS_NO_MEMORY;

	result = hsakmt_validate_nodeid(NodeId, &gpu_id);
	if (result != HSAKMT_STATUS_SUCCESS)
		return result;

	struct kfd_ioctl_dbg_register_args args = {0};

	args.gpu_id = gpu_id;

	long err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_DBG_REGISTER_DEPRECATED, &args);

	if (err == 0)
		result = HSAKMT_STATUS_SUCCESS;
	else
		result = HSAKMT_STATUS_ERROR;

	return result;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDbgUnregister(HSAuint32 NodeId)
{
	uint32_t gpu_id;
	HSAKMT_STATUS result;

	CHECK_KFD_OPEN();

	if (!is_device_debugged)
		return HSAKMT_STATUS_NO_MEMORY;

	result = hsakmt_validate_nodeid(NodeId, &gpu_id);
	if (result != HSAKMT_STATUS_SUCCESS)
		return result;

	struct kfd_ioctl_dbg_unregister_args args = {0};

	args.gpu_id = gpu_id;
	long err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_DBG_UNREGISTER_DEPRECATED, &args);

	if (err)
		return HSAKMT_STATUS_ERROR;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDbgWavefrontControl(HSAuint32 NodeId,
						  HSA_DBG_WAVEOP Operand,
						  HSA_DBG_WAVEMODE Mode,
						  HSAuint32 TrapId,
						  HsaDbgWaveMessage *DbgWaveMsgRing)
{
	HSAKMT_STATUS result;
	uint32_t gpu_id;

	struct kfd_ioctl_dbg_wave_control_args *args;

	CHECK_KFD_OPEN();

	result = hsakmt_validate_nodeid(NodeId, &gpu_id);
	if (result != HSAKMT_STATUS_SUCCESS)
		return result;


/* Determine Size of the ioctl buffer */
	uint32_t buff_size = sizeof(Operand) + sizeof(Mode) + sizeof(TrapId) +
			     sizeof(DbgWaveMsgRing->DbgWaveMsg) +
			     sizeof(DbgWaveMsgRing->MemoryVA) + sizeof(*args);

	args = (struct kfd_ioctl_dbg_wave_control_args *)malloc(buff_size);
	if (!args)
		return HSAKMT_STATUS_ERROR;

	memset(args, 0, buff_size);

	args->gpu_id = gpu_id;
	args->buf_size_in_bytes = buff_size;

	/* increment pointer to the start of the non fixed part */
	unsigned char *run_ptr = (unsigned char *)args + sizeof(*args);

	/* save variable content pointer for kfd */
	args->content_ptr = (uint64_t)run_ptr;

	/* insert items, and increment pointer accordingly */
	*((HSA_DBG_WAVEOP *)run_ptr) = Operand;
	run_ptr += sizeof(Operand);

	*((HSA_DBG_WAVEMODE *)run_ptr) = Mode;
	run_ptr += sizeof(Mode);

	*((HSAuint32 *)run_ptr) = TrapId;
	run_ptr += sizeof(TrapId);

	*((HsaDbgWaveMessageAMD *)run_ptr) = DbgWaveMsgRing->DbgWaveMsg;
	run_ptr += sizeof(DbgWaveMsgRing->DbgWaveMsg);

	*((void **)run_ptr) = DbgWaveMsgRing->MemoryVA;
	run_ptr += sizeof(DbgWaveMsgRing->MemoryVA);

	/* send to kernel */
	long err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_DBG_WAVE_CONTROL_DEPRECATED, args);

	free(args);

	if (err)
		return HSAKMT_STATUS_ERROR;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDbgAddressWatch(HSAuint32 NodeId,
					      HSAuint32 NumWatchPoints,
					      HSA_DBG_WATCH_MODE WatchMode[],
					      void *WatchAddress[],
					      HSAuint64 WatchMask[],
					      HsaEvent *WatchEvent[])
{
	HSAKMT_STATUS result;
	uint32_t gpu_id;

	/* determine the size of the watch mask and event buffers
	 * the value is NULL if and only if no vector data should be attached
	 */
	uint32_t watch_mask_items = WatchMask[0] > 0 ? NumWatchPoints:1;
	uint32_t watch_event_items = WatchEvent != NULL ? NumWatchPoints:0;

	struct kfd_ioctl_dbg_address_watch_args *args;
	HSAuint32		 i = 0;

	CHECK_KFD_OPEN();

	result = hsakmt_validate_nodeid(NodeId, &gpu_id);
	if (result != HSAKMT_STATUS_SUCCESS)
		return result;

	if (NumWatchPoints > MAX_ALLOWED_NUM_POINTS)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	/* Size and structure of the ioctl buffer is dynamic in this case
	 * Here we calculate the buff size.
	 */
	uint32_t buff_size = sizeof(NumWatchPoints) +
		(sizeof(WatchMode[0]) + sizeof(WatchAddress[0])) *
			NumWatchPoints +
		watch_mask_items * sizeof(HSAuint64) +
		watch_event_items * sizeof(HsaEvent *) + sizeof(*args);

	args = (struct kfd_ioctl_dbg_address_watch_args *) malloc(buff_size);
	if (!args)
		return HSAKMT_STATUS_ERROR;

	memset(args, 0, buff_size);

	args->gpu_id = gpu_id;
	args->buf_size_in_bytes = buff_size;


	/* increment pointer to the start of the non fixed part */
	unsigned char *run_ptr = (unsigned char *)args + sizeof(*args);

	/* save variable content pointer for kfd */
	args->content_ptr = (uint64_t)run_ptr;
	/* insert items, and increment pointer accordingly */

	*((HSAuint32 *)run_ptr) = NumWatchPoints;
	run_ptr += sizeof(NumWatchPoints);

	for (i = 0; i < NumWatchPoints; i++) {
		*((HSA_DBG_WATCH_MODE *)run_ptr) = WatchMode[i];
		run_ptr += sizeof(WatchMode[i]);
	}

	for (i = 0; i < NumWatchPoints; i++) {
		*((void **)run_ptr) = WatchAddress[i];
		run_ptr += sizeof(WatchAddress[i]);
	}

	for (i = 0; i < watch_mask_items; i++) {
		*((HSAuint64 *)run_ptr) = WatchMask[i];
		run_ptr += sizeof(WatchMask[i]);
	}

	for (i = 0; i < watch_event_items; i++)	{
		*((HsaEvent **)run_ptr) = WatchEvent[i];
		run_ptr += sizeof(WatchEvent[i]);
	}

	/* send to kernel */
	long err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_DBG_ADDRESS_WATCH_DEPRECATED, args);

	free(args);

	if (err)
		return HSAKMT_STATUS_ERROR;
	return HSAKMT_STATUS_SUCCESS;
}

#define HSA_RUNTIME_ENABLE_MAX_MAJOR   1
#define HSA_RUNTIME_ENABLE_MIN_MINOR   13

HSAKMT_STATUS HSAKMTAPI hsaKmtCheckRuntimeDebugSupport(void) {
	HsaNodeProperties node = {0};
	HsaSystemProperties props = {0};
	HsaVersionInfo versionInfo = {0};

	memset(&node, 0x00, sizeof(node));
	memset(&props, 0x00, sizeof(props));
	if (hsaKmtAcquireSystemProperties(&props))
		return HSAKMT_STATUS_ERROR;

	//the firmware of gpu node doesn't support the debugger, disable it.
	for (uint32_t i = 0; i < props.NumNodes; i++) {
		if (hsaKmtGetNodeProperties(i, &node))
			return HSAKMT_STATUS_ERROR;

		//ignore cpu node
		if (node.NumCPUCores && !node.NumFComputeCores)
			continue;
		if (!node.Capability.ui32.DebugSupportedFirmware)
			return HSAKMT_STATUS_NOT_SUPPORTED;
	}

	if (hsaKmtGetVersion(&versionInfo))
		return HSAKMT_STATUS_NOT_SUPPORTED;

	if (versionInfo.KernelInterfaceMajorVersion < HSA_RUNTIME_ENABLE_MAX_MAJOR ||
		(versionInfo.KernelInterfaceMajorVersion ==
			HSA_RUNTIME_ENABLE_MAX_MAJOR &&
		(int)versionInfo.KernelInterfaceMinorVersion < HSA_RUNTIME_ENABLE_MIN_MINOR))
		return HSAKMT_STATUS_NOT_SUPPORTED;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRuntimeEnable(void *rDebug,
					    bool setupTtmp)
{
	struct kfd_ioctl_runtime_enable_args args = {0};
	HSAKMT_STATUS result = hsaKmtCheckRuntimeDebugSupport();

	if (result)
		return result;

	memset(&args, 0x00, sizeof(args));
	args.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK |
		((setupTtmp) ? KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK : 0);
	args.r_debug = (HSAuint64)rDebug;

	long err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_RUNTIME_ENABLE, &args);

	if (err) {
		if (errno == EBUSY)
			return HSAKMT_STATUS_UNAVAILABLE;
		else
			return HSAKMT_STATUS_ERROR;
	}
	runtime_capabilities_mask= args.capabilities_mask;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRuntimeDisable(void)
{
	struct kfd_ioctl_runtime_enable_args args = {0};
	HSAKMT_STATUS result = hsaKmtCheckRuntimeDebugSupport();

	if (result)
		return result;

	memset(&args, 0x00, sizeof(args));
	args.mode_mask = 0; //Disable

	if (hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_RUNTIME_ENABLE, &args))
		return HSAKMT_STATUS_ERROR;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetRuntimeCapabilities(HSAuint32 *caps_mask)
{
	*caps_mask = runtime_capabilities_mask;
	return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS dbg_trap_get_device_data(void *data,
					      uint32_t *n_entries,
					      uint32_t entry_size)
{
	struct kfd_ioctl_dbg_trap_args args = {0};

	args.device_snapshot.snapshot_buf_ptr = (uint64_t) data;
	args.device_snapshot.num_devices = *n_entries;
	args.device_snapshot.entry_size = entry_size;
	args.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
	args.pid = getpid();
	if (hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_DBG_TRAP, &args))
		return HSAKMT_STATUS_ERROR;
	*n_entries = args.device_snapshot.num_devices;

	return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS dbg_trap_get_queue_data(void *data,
					     uint32_t *n_entries,
					     uint32_t entry_size,
					     uint32_t *queue_ids)
{
	struct kfd_ioctl_dbg_trap_args args = {0};

	args.queue_snapshot.num_queues = *n_entries;
	args.queue_snapshot.entry_size = entry_size;
	args.queue_snapshot.exception_mask = KFD_EC_MASK(EC_QUEUE_NEW);
	args.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
	args.queue_snapshot.snapshot_buf_ptr = (uint64_t) data;
	args.pid = getpid();

	if (hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_DBG_TRAP, &args))
		return HSAKMT_STATUS_ERROR;

	*n_entries = args.queue_snapshot.num_queues;
	if (queue_ids && *n_entries) {
		struct kfd_queue_snapshot_entry *queue_entry =
		    (struct kfd_queue_snapshot_entry *) data;
		for (uint32_t i = 0; i < *n_entries; i++)
			queue_ids[i] = queue_entry[i].queue_id;
	}

	return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS dbg_trap_suspend_queues(uint32_t *queue_ids,
					     uint32_t num_queues)
{
	struct kfd_ioctl_dbg_trap_args args = {0};
	int r;

	args.suspend_queues.queue_array_ptr = (uint64_t) queue_ids;
	args.suspend_queues.num_queues = num_queues;
	args.suspend_queues.exception_mask = KFD_EC_MASK(EC_QUEUE_NEW);
	args.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
	args.pid = getpid();

	r = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_DBG_TRAP, &args);
	if (r < 0)
		return HSAKMT_STATUS_ERROR;

	return HSAKMT_STATUS_SUCCESS;
}

/* Debugger support has been in KFD ABI 1.13.  */
#define KFD_MINOR_MIN_DEBUG 13

HSAKMT_STATUS HSAKMTAPI hsaKmtDbgEnable(void **runtime_info,
					     HSAuint32 *data_size)
{
	struct kfd_ioctl_dbg_trap_args args = {0};

	CHECK_KFD_OPEN();
	CHECK_KFD_MINOR_VERSION(KFD_MINOR_MIN_DEBUG);
	*data_size = sizeof(struct kfd_runtime_info);
	args.enable.rinfo_size = *data_size;
	args.enable.dbg_fd = hsakmt_kfd_fd;
	*runtime_info = malloc(args.enable.rinfo_size);
	if (!*runtime_info)
		return HSAKMT_STATUS_NO_MEMORY;
	args.enable.rinfo_ptr = (uint64_t) *runtime_info;
	args.op = KFD_IOC_DBG_TRAP_ENABLE;
	args.pid = getpid();

	if (hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_DBG_TRAP, &args)) {
		free(*runtime_info);
		return HSAKMT_STATUS_ERROR;
	}

	return HSAKMT_STATUS_SUCCESS;
}
HSAKMT_STATUS HSAKMTAPI hsaKmtDbgDisable(void)
{
	struct kfd_ioctl_dbg_trap_args args = {0};

	CHECK_KFD_OPEN();
	CHECK_KFD_MINOR_VERSION(KFD_MINOR_MIN_DEBUG);
	args.enable.dbg_fd = hsakmt_kfd_fd;
	args.op = KFD_IOC_DBG_TRAP_DISABLE;
	args.pid = getpid();

	if (hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_DBG_TRAP, &args))
		return HSAKMT_STATUS_ERROR;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDbgGetDeviceData(void **data,
						HSAuint32 *n_entries,
						HSAuint32 *entry_size)
{
	HSAKMT_STATUS ret = HSAKMT_STATUS_NO_MEMORY;

	CHECK_KFD_OPEN();
	CHECK_KFD_MINOR_VERSION(KFD_MINOR_MIN_DEBUG);
	*n_entries = UINT32_MAX;
	*entry_size = sizeof(struct kfd_dbg_device_info_entry);
	*data = malloc(*entry_size * *n_entries);
	if (!*data)
		return ret;
	ret = dbg_trap_get_device_data(*data, n_entries, *entry_size);
	if (ret)
		free(*data);

	return ret;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDbgGetQueueData(void **data,
						HSAuint32 *n_entries,
						HSAuint32 *entry_size,
						bool suspend_queues)
{
	uint32_t *queue_ids = NULL;

	CHECK_KFD_OPEN();
	CHECK_KFD_MINOR_VERSION(KFD_MINOR_MIN_DEBUG);
	*entry_size = sizeof(struct kfd_queue_snapshot_entry);
	*n_entries = 0;
	if (dbg_trap_get_queue_data(NULL, n_entries, *entry_size, NULL))
		return HSAKMT_STATUS_ERROR;
	*data = malloc(*n_entries * *entry_size);
	if (!*data)
		return HSAKMT_STATUS_NO_MEMORY;
	if (suspend_queues && *n_entries)
		queue_ids = (uint32_t *)malloc(sizeof(uint32_t) * *n_entries);
	if (!queue_ids ||
	    dbg_trap_get_queue_data(*data, n_entries, *entry_size, queue_ids))
		goto free_data;
	if (queue_ids) {
		if (dbg_trap_suspend_queues(queue_ids, *n_entries) ||
		    dbg_trap_get_queue_data(*data, n_entries, *entry_size, NULL))
			goto free_data;
		free(queue_ids);
	}
	return HSAKMT_STATUS_SUCCESS;
free_data:
	free(*data);
	if (queue_ids)
		free(queue_ids);

	return HSAKMT_STATUS_ERROR;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDebugTrapIoctl(struct kfd_ioctl_dbg_trap_args *args,
					HSA_QUEUEID *Queues,
					HSAuint64 *DebugReturn)
{
	HSAKMT_STATUS result;

	CHECK_KFD_OPEN();

	if (Queues) {
		int num_queues = args->op == KFD_IOC_DBG_TRAP_SUSPEND_QUEUES ?
						args->suspend_queues.num_queues :
						args->resume_queues.num_queues;
		void *queue_ptr = args->op == KFD_IOC_DBG_TRAP_SUSPEND_QUEUES ?
						(void *)args->suspend_queues.queue_array_ptr :
						(void *)args->resume_queues.queue_array_ptr;

		uint32_t *queue_ids = hsakmt_convert_queue_ids(num_queues, Queues);
		if (!queue_ids) {
			return HSAKMT_STATUS_NO_MEMORY;
		}
		memcpy(queue_ptr, queue_ids, num_queues * sizeof(uint32_t));
		free(queue_ids);
	}

	long err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_DBG_TRAP, args);
	if (DebugReturn)
		*DebugReturn = err;

	if (args->op == KFD_IOC_DBG_TRAP_SUSPEND_QUEUES &&
				err >= 0 && err <= args->suspend_queues.num_queues)
		result = HSAKMT_STATUS_SUCCESS;
	else if (args->op == KFD_IOC_DBG_TRAP_RESUME_QUEUES &&
				err >= 0 && err <= args->resume_queues.num_queues)
		result = HSAKMT_STATUS_SUCCESS;
	else if (err == 0)
		result = HSAKMT_STATUS_SUCCESS;
	else
		result = HSAKMT_STATUS_ERROR;

	return result;
}
