/*
 * Copyright © 2025 Advanced Micro Devices, Inc.
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

#include "hsakmt/hsakmtmodel.h"
#include "libhsakmt.h"
#include "hsakmt/hsakmttypes.h"
#include "hsakmt/hsakmtmodeliface.h"
#define _GNU_SOURCE
#define __USE_GNU
#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <fcntl.h>

bool hsakmt_use_model;
char *hsakmt_model_topology;

struct model_node
{
	bool is_gpu;
	void *aperture;
	hsakmt_model_t *model;
	uint64_t doorbell_offset;
	uint64_t total_memory_size;
	uint64_t allocated_memory_size;
};

struct model_event
{
	uint32_t event_type;
	uint32_t auto_reset;
	uint64_t value;
};

struct model_mem_data
{
	uint64_t va_addr;
	uint64_t file_offset;
	uint64_t size;
	uint64_t mapped_nodes_bitmask;
	uint32_t flags;
	uint32_t node_id;
};

struct model_queue
{
	hsakmt_model_queue_t *queue;
	uint32_t node_id;
};

#define MAX_MODEL_QUEUES 128
// Use a 256GB aperture for the model.
#define MODEL_APERTURE_SIZE (1llu << 38)
static void *model_mmio_page;
static pthread_mutex_t model_ioctl_mutex = PTHREAD_MUTEX_INITIALIZER;
static unsigned model_event_limit;
static uint64_t *model_event_bitmap;
static struct model_event *model_events;
static pthread_cond_t model_event_condvar;
static void *model_library;
static const struct hsakmt_model_functions *model_functions;
static uint64_t model_memfd_size;
static uint64_t model_num_nodes;
static struct model_node *model_nodes;
static struct model_queue model_queues[MAX_MODEL_QUEUES];

HSAKMT_STATUS HSAKMTAPI hsaKmtModelEnabled(bool* enable)
{
	*enable = hsakmt_use_model;
	return HSAKMT_STATUS_SUCCESS;
}

void model_init_env_vars(void)
{
	/* Check whether to use a model instead of real hardware */
	hsakmt_model_topology = getenv("HSA_MODEL_TOPOLOGY");
	if (hsakmt_model_topology)
		hsakmt_use_model = true;
	if (hsakmt_use_model)
	{
		/* Backing memory file is used to stand in for the kfd_fd,
		 * which is needed early, so create it already.
		 *
		 * For old systems without memfd_create, or if the user prefers,
		 * we create a regular backing file. Prefer to use memfd_create
		 * by default where possible.
		 */
		int fd = -1;
		const char *fname = getenv("HSA_MODEL_MEMFILE");
		if (fname)
		{
			fprintf(stderr, "model: use memory backing file given in HSA_MODEL_MEMFILE: %s\n", fname);

			fd = open(fname, O_CREAT | O_EXCL | O_CLOEXEC | O_RDWR, S_IRUSR | S_IWUSR);
			if (fd < 0)
			{
				perror("model: failed to create backing file");
				abort();
			}

			unlink(fname);
		}

		if (fd < 0)
		{
#ifdef HAVE_MEMFD_CREATE
			fd = memfd_create("hsakmt_model", MFD_CLOEXEC);
			if (fd < 0)
			{
				fprintf(stderr, "model: Failed to create memfd\n");
				abort();
			}
#else
			fprintf(stderr, "model: built without memfd support\n"
							"model: set HSA_MODEL_MEMFILE to path of a backing file\n");
			abort();
#endif
		}
		assert(hsakmt_kfd_fd < 0);
		hsakmt_kfd_fd = fd;
		pthread_condattr_t condattr;
		pthread_condattr_init(&condattr);
		pthread_condattr_setclock(&condattr, CLOCK_MONOTONIC);
		pthread_cond_init(&model_event_condvar, &condattr);
		pthread_condattr_destroy(&condattr);
		const char *libname = getenv("HSA_MODEL_LIB");
		if (!libname)
		{
			fprintf(stderr, "model: HSA_MODEL_LIB environment variable must be set to FFM .so\n");
			abort();
		}
		// model_library = dlmopen(LM_ID_NEWLM, libname, RTLD_NOW);
		model_library = dlopen(libname, RTLD_NOW | RTLD_LOCAL);
		if (!model_library)
		{
			fprintf(stderr, "model: failed to load %s: %s\n", libname, dlerror());
			abort();
		}
		get_hsakmt_model_functions_t getter = dlsym(model_library, "get_hsakmt_model_functions");
		if (!getter)
		{
			fprintf(stderr, "model: Failed to get hsakmt_model_functions\n");
			abort();
		}
		model_functions = getter();
		if (model_functions->version_major != HSAKMT_MODEL_INTERFACE_VERSION_MAJOR ||
			model_functions->version_minor < HSAKMT_MODEL_INTERFACE_VERSION_MINOR)
		{
			fprintf(stderr, "model: Model has interface version %u.%u, need version %u.%u\n",
					model_functions->version_major, model_functions->version_minor,
					HSAKMT_MODEL_INTERFACE_VERSION_MAJOR, HSAKMT_MODEL_INTERFACE_VERSION_MINOR);
			abort();
		}
	}
}

static uint64_t allocate_from_memfd(uint64_t size, uint64_t align)
{
	if (!align)
		align = 4096;
	assert(POWER_OF_2(align)); /* must be power of two */
	assert(align >= 4096);
	size = (size + 4095) & ~4095;
	model_memfd_size = (model_memfd_size + align - 1) & ~(align - 1);
	uint64_t offset = model_memfd_size;
	model_memfd_size += size;
	int ret = ftruncate(hsakmt_kfd_fd, model_memfd_size);
	if (ret < 0)
	{
		fprintf(stderr, "model: ftruncate on memfd failed\n");
		abort();
	}
	return offset;
}
static uint64_t get_sysfs_mem_bank_size(unsigned node_id, unsigned mem_id)
{
	char prop_name[256];
	char path[256];
	snprintf(path, sizeof(path), "%s/nodes/%u/mem_banks/%u/properties",
			 hsakmt_model_topology, node_id, mem_id);
	FILE *f = fopen(path, "r");
	if (!f)
	{
		fprintf(stderr, "model: Failed to open %s\n", path);
		abort();
	}
	uint64_t prop_val;
	while (fscanf(f, "%s %" PRIu64 "\n", prop_name, &prop_val) == 2)
	{
		if (!strcmp(prop_name, "size_in_bytes"))
		{
			fclose(f);
			return prop_val;
		}
	}
	fprintf(stderr, "model: Missing size_in_bytes in %s\n", path);
	abort();
}

static void model_set_event(void *data, unsigned event_id)
{
	if (!event_id)
		return;

	if (event_id > model_event_limit)
	{
		fprintf(stderr, "model_set_event: event_id = %u out of bounds\n",
				event_id);
		abort();
	}

	unsigned slot = event_id - 1;

	if (!((model_event_bitmap[slot / 64] >> (slot % 64)) & 1))
	{
		fprintf(stderr, "model_set_event: event_id = %u is not allocated\n",
				event_id);
		abort();
	}

	struct model_event *event = &model_events[slot];
	if (event->event_type == HSA_EVENTTYPE_SIGNAL)
	{
		assert(model_events[slot].value <= 1);
		model_events[slot].value = 1;
	}
	else
	{
		fprintf(stderr, "model: Unimplemented event type\n");
		abort();
	}

	pthread_cond_broadcast(&model_event_condvar);
}

void model_init(void)
{
	if (!hsakmt_use_model)
		return;
	HSAKMT_STATUS result;
	HsaSystemProperties props;
	/* Read the topology to determine nodes. */
	result = hsakmt_topology_sysfs_get_system_props(&props);
	if (result != HSAKMT_STATUS_SUCCESS)
	{
		fprintf(stderr, "model: Failed to parse topology\n");
		abort();
	}
	model_nodes = calloc(props.NumNodes, sizeof(*model_nodes));
	if (!model_nodes)
		abort();
	model_num_nodes = props.NumNodes;
	for (unsigned node_id = 0; node_id < props.NumNodes; node_id++)
	{
		HsaNodeProperties node_props;
		result = hsakmt_topology_get_node_props(node_id, &node_props);
		if (result != HSAKMT_STATUS_SUCCESS)
		{
			fprintf(stderr, "model: Failed to get node %u properties\n", node_id);
			abort();
		}
		if (node_props.KFDGpuID == 0)
			continue;
		if (node_props.KFDGpuID != node_id + 1)
		{
			fprintf(stderr,
					"model: Node %u has KFD GPU ID %u, but should be %u."
					" Please change the gpu_id file.\n",
					node_id, node_props.KFDGpuID, node_id + 1);
			abort();
		}
		model_nodes[node_id].is_gpu = true;
		/* Reserve the VA space for the aperture, but don't fill it with pages. */
		model_nodes[node_id].aperture =
			mmap(NULL, MODEL_APERTURE_SIZE, PROT_NONE,
				 MAP_PRIVATE | MAP_NORESERVE | MAP_ANONYMOUS, -1, 0);
		pr_debug("Modeling Creating Memory Aperture: %p\n", model_nodes[node_id].aperture);
		if (model_nodes[node_id].aperture == MAP_FAILED)
		{
			fprintf(stderr, "model: Failed to reserve aperture via mmap\n");
			abort();
		}
		/* Create the doorbell region */
		model_nodes[node_id].doorbell_offset = allocate_from_memfd(8192, 8192);
		for (unsigned mem_id = 0; mem_id < node_props.NumMemoryBanks; ++mem_id)
		{
			model_nodes[node_id].total_memory_size += get_sysfs_mem_bank_size(node_id, mem_id);
		}
		/* Create the model */
		// TODO: Move this into a separate thread
		model_nodes[node_id].model = model_functions->create();
		if (!model_nodes[node_id].model)
		{
			fprintf(stderr, "model: Failed to create model\n");
			abort();
		}
		model_functions->set_global_aperture(model_nodes[node_id].model,
											 model_nodes[node_id].aperture,
											 MODEL_APERTURE_SIZE);

		model_functions->set_set_event(model_nodes[node_id].model, model_set_event, NULL);
	}
}
void model_set_mmio_page(void *ptr)
{
	assert(!model_mmio_page);
	model_mmio_page = ptr;
}
void model_set_event_page(void *ptr, unsigned event_limit)
{
	// TODO: Fully understand what's happening with this page and the event limit.
	//       ROCR-Runtime allocates a pool of 4096 events, but also a handful or so
	//       of additional events, which blows through the event_limit of 4096
	//       that is passed here. And it seems that not using the page at all
	//       is supported?
	assert(!model_event_limit);
	assert(event_limit % 64 == 0);
	event_limit *= 2;
	model_event_limit = event_limit;
	model_event_bitmap = calloc(event_limit / 64, 8);
	model_events = calloc(event_limit, sizeof(*model_events));
}
/* Model implementation of KFD ioctl. */

static int model_kfd_ioctl_locked(unsigned long request, void *arg)
{
	assert(_IOC_TYPE(request) == AMDKFD_IOCTL_BASE);
	if (_IOC_NR(request) == 0x20)
	{
		// This is AMDKFD_IOC_SVM. It is defined / used in an unusual way.
		struct kfd_ioctl_svm_args *args = arg;
		if (args->op == KFD_IOCTL_SVM_OP_SET_ATTR)
		{
			// todo?
			return 0;
		}
		fprintf(stderr, "model: Unimplemented SVM op\n");
		abort();
	}
	switch (request)
	{
	case AMDKFD_IOC_GET_VERSION:
	{
		pr_debug("MODEL IOCTL: AMDKFD_IOC_GET_VERSION\n");
		struct kfd_ioctl_get_version_args *args = arg;
		args->major_version = 1;
		args->minor_version = 14;
		return 0;
	}
	case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
	{
		pr_debug("MODEL IOCTL: AMDKFD_IOC_GET_PROCESS_APERTURES_NEW\n");
		struct kfd_ioctl_get_process_apertures_new_args *args = arg;
		struct kfd_process_device_apertures *apertures =
			(void *)args->kfd_process_device_apertures_ptr;
		assert(args->num_of_nodes == model_num_nodes);
		for (unsigned node_id = 0; node_id < args->num_of_nodes; ++node_id)
		{
			memset(&apertures[node_id], 0, sizeof(apertures[node_id]));
			if (!model_nodes[node_id].is_gpu)
				continue;
			apertures[node_id].gpu_id = 1 + node_id;
			apertures[node_id].gpuvm_base = 0x4000llu;
			apertures[node_id].gpuvm_limit = MODEL_APERTURE_SIZE;
			apertures[node_id].lds_base = 0x4000000000000000llu; // 0x1000000000000?
			apertures[node_id].lds_limit = 0x40000000ffffffffllu;
			apertures[node_id].scratch_base = 0x5000000000000000llu; // 0x2000000000000?
			apertures[node_id].scratch_limit = 0x50000000ffffffffllu;
		}
		return 0;
	}
	case AMDKFD_IOC_SET_XNACK_MODE:
	{
		pr_debug("MODEL IOCTL: AMDKFD_IOC_SET_XNACK_MODE\n");
		// Don't support XNACK
		struct kfd_ioctl_set_xnack_mode_args *args = arg;
		if (args->xnack_enabled < 0)
		{
			args->xnack_enabled = 0;
			return 0;
		}
		errno = EPERM;
		return -1;
	}
	case AMDKFD_IOC_GET_CLOCK_COUNTERS:
	{
		pr_debug("MODEL IOCTL: AMDKFD_IOC_GET_CLOCK_COUNTERS\n");
		struct kfd_ioctl_get_clock_counters_args *args = arg;
		args->gpu_clock_counter = 0; // TODO
		args->cpu_clock_counter = 0;
		args->system_clock_counter = 0;
		args->system_clock_freq = 0;
		return 0;
	}
	case AMDKFD_IOC_ACQUIRE_VM:
		pr_debug("MODEL IOCTL: AMDKFD_IOC_ACQUIRE_VM\n");
		return 0;
	case AMDKFD_IOC_SET_MEMORY_POLICY:
	{
		pr_debug("MODEL IOCTL: AMDKFD_IOC_SET_MEMORY_POLICY\n");
		// todo?
		return 0;
	}
	case AMDKFD_IOC_AVAILABLE_MEMORY:
	{
		pr_debug("MODEL IOCTL: AMDKFD_IOC_AVAILABLE_MEMORY\n");
		static const uint64_t minimum_reported = 128 * 1024 * 1024;
		struct kfd_ioctl_get_available_memory_args *args = arg;
		unsigned node_id = args->gpu_id - 1;
		struct model_node *node = &model_nodes[node_id];
		assert(node_id < model_num_nodes);
		if (node->allocated_memory_size + minimum_reported >= node->total_memory_size)
			args->available = minimum_reported;
		else
			args->available = node->total_memory_size - node->allocated_memory_size;
		return 0;
	}
	case AMDKFD_IOC_ALLOC_MEMORY_OF_GPU:
	{
		// Expect an SVM style allocation: The memory is allocated on the host
		// side e.g. via mmap(), and this IOCTL "only" registers the memory
		// with the GPU. This is a no-op for us because we aren't a GPU.
		struct kfd_ioctl_alloc_memory_of_gpu_args *args = arg;
		unsigned node_id = args->gpu_id - 1;
		assert(node_id < model_num_nodes);
		assert(model_nodes[node_id].is_gpu);
		if (args->va_addr == 0)
		{
			fprintf(stderr, "model: Expect only SVM allocations?\n");
			abort();
		}
		if (args->size % PAGE_SIZE != 0)
		{
			fprintf(stderr, "model: Allocation size not a multiple of page size\n");
			abort();
		}
		if (args->flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR)
		{
			fprintf(stderr, "model: userptr not supported\n");
			abort();
		}
		struct model_mem_data *mem_data = calloc(1, sizeof(*mem_data));
		if (!mem_data)
			abort();
		mem_data->va_addr = args->va_addr;
		mem_data->size = args->size;
		mem_data->flags = args->flags;
		mem_data->node_id = node_id;
		if (args->flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL)
		{
			assert(args->size == 8192);
			mem_data->file_offset = model_nodes[node_id].doorbell_offset;
		}
		else
		{
			mem_data->file_offset = allocate_from_memfd(args->size, 0);
		}
		args->handle = (__u64)mem_data;
		args->mmap_offset = mem_data->file_offset;
		model_nodes[node_id].allocated_memory_size += args->size;
		pr_debug("MODEL IOCTL: AMDKFD_IOC_ALLOC_MEMORY_OF_GPU: VA: %lx : Size: %lu, Flags: %x\n", mem_data->va_addr, mem_data->size, mem_data->flags);
		model_functions->alloced_memory(model_nodes[node_id].model, (uint64_t *)mem_data->va_addr, mem_data->size, mem_data->flags);
		return 0;
	}
	case AMDKFD_IOC_FREE_MEMORY_OF_GPU:
	{
		struct kfd_ioctl_free_memory_of_gpu_args *args = arg;
		struct model_mem_data *mem_data = (void *)args->handle;
		assert(!mem_data->mapped_nodes_bitmask);
		// Free the memory by punching a hole into the underlying memfd.
		//
		// Ideally, we'd also remember holes in the file and re-use them for
		// allocations to avoid the file size from growing indefinitely. It's
		// unclear whether the current implementation causes kernel data
		// structures to grow. But in practice, it almost certainly never
		// matters.
		int ret = fallocate(hsakmt_kfd_fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
							mem_data->file_offset, mem_data->size);
		if (ret != 0)
		{
			perror("model: failed to punch hole in memfd");
			abort();
		}
		model_nodes[mem_data->node_id].allocated_memory_size -= mem_data->size;
		model_functions->freed_memory(model_nodes[mem_data->node_id].model, (uint64_t *)mem_data->va_addr, mem_data->size);
		pr_debug("MODEL IOCTL: AMDKFD_IOC_FREE_MEMORY_OF_GPU: VA: %lx : Size: %lu, Flags: %x\n", mem_data->va_addr, mem_data->size, mem_data->flags);
		free(mem_data);
		return 0;
	}
	case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
	{
		struct kfd_ioctl_map_memory_to_gpu_args *args = arg;
		struct model_mem_data *mem_data = (void *)args->handle;
		while (args->n_success < args->n_devices)
		{
			uint32_t gpu_id = ((uint32_t *)args->device_ids_array_ptr)[args->n_success];
			uint32_t node_id = gpu_id - 1;
			assert(node_id < model_num_nodes);
			if (mem_data->mapped_nodes_bitmask & (1llu << node_id))
			{
				fprintf(stderr, "model: Already mapped\n");
				abort();
			}
			assert(model_nodes[node_id].aperture);
			unsigned prot = PROT_READ;
			if (mem_data->flags & KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE)
				prot |= PROT_WRITE;
			// TODO: Mark *shader*-executable memory?

			pr_debug("MODEL IOCTL: AMDKFD_IOC_MAP_MEMORY_TO_GPU: VA: %lx : Size: %lu, Flags: %x\n", mem_data->va_addr, mem_data->size, mem_data->flags);
			void *ret = mmap(VOID_PTR_ADD(model_nodes[node_id].aperture, mem_data->va_addr),
							 mem_data->size, prot,
							 MAP_SHARED | MAP_FIXED, hsakmt_kfd_fd, mem_data->file_offset);
			if (ret == MAP_FAILED)
			{
				fprintf(stderr, "model: mmap failed\n");
				abort();
			}
			mem_data->mapped_nodes_bitmask |= (1llu << node_id);
			args->n_success++;
		}
		return 0;
	}
	case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
	{
		pr_debug("MODEL IOCTL: AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU\n");
		struct kfd_ioctl_unmap_memory_from_gpu_args *args = arg;
		struct model_mem_data *mem_data = (void *)args->handle;
		while (args->n_success < args->n_devices)
		{
			uint32_t gpu_id = ((uint32_t *)args->device_ids_array_ptr)[args->n_success];
			uint32_t node_id = gpu_id - 1;
			assert(node_id < model_num_nodes);
			if (!(mem_data->mapped_nodes_bitmask & (1llu << node_id)))
			{
				fprintf(stderr, "model: Not mapped\n");
				abort();
			}
			assert(model_nodes[node_id].aperture);
			/* Overwrite the mapping with an empty mapping to keep
			 * it reserved. */
			void *ret = mmap(VOID_PTR_ADD(model_nodes[node_id].aperture, mem_data->va_addr),
							 mem_data->size, PROT_NONE,
							 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
			if (ret == MAP_FAILED)
			{
				perror("model: unmap failed");
				abort();
			}
			mem_data->mapped_nodes_bitmask &= ~(1llu << node_id);
			args->n_success++;
		}
		args->n_success = args->n_devices;
		return 0;
	}
	case AMDKFD_IOC_CREATE_EVENT:
	{
		struct kfd_ioctl_create_event_args *args = arg;
		pr_debug("MODEL IOCTL: AMDKFD_IOC_CREATE_EVENT: %u\n", args->event_type);
		// Find a free slot
		unsigned i;
		for (i = 0; i < model_event_limit; i += 64)
		{
			uint64_t bitmap = model_event_bitmap[i / 64];
			if (bitmap == ~(uint64_t)0)
				continue;
			i += ffsll(~bitmap) - 1;
			break;
		}
		if (i >= model_event_limit)
		{
			fprintf(stderr, "model: Ran out of event slots. Should be an application error.\n");
			abort();
		}
		// Allocate the signal
		model_event_bitmap[i / 64] |= (uint64_t)1 << (i % 64);
		model_events[i].event_type = args->event_type;
		model_events[i].auto_reset = args->auto_reset;
		model_events[i].value = 0;
		args->event_trigger_data = 0xbadf001; // ???
		args->event_id = 1 + i;
		args->event_slot_index = ~0;
		return 0;
	}
	case AMDKFD_IOC_WAIT_EVENTS:
	{
		struct kfd_ioctl_wait_events_args *args = arg;
		struct kfd_event_data *events = (void *)args->events_ptr;
		pr_debug("MODEL IOCTL: AMDKFD_IOC_WAIT_EVENTS: %u\n", args->num_events);
		bool have_timeout = args->timeout != 0xffffffffu;
		bool hit_timeout = false;
		struct timespec timeout;
		if (have_timeout)
		{
			clock_gettime(CLOCK_MONOTONIC, &timeout);
			timeout.tv_sec += args->timeout / 1000;
			timeout.tv_nsec += (args->timeout % 1000) * 1000000;
			if (timeout.tv_nsec > 1000000000)
			{
				timeout.tv_nsec -= 1000000000;
				timeout.tv_sec++;
			}
		}
		for (;;)
		{
			bool final_ready = args->wait_for_all;
			for (unsigned i = 0; i < args->num_events; ++i)
			{
				unsigned slot = events[i].event_id - 1;
				struct model_event *event = &model_events[slot];
				bool this_ready = false;
				if (event->event_type == HSA_EVENTTYPE_SIGNAL)
				{
					uint64_t current_age = event->value;
					uint64_t target_age = events[i].signal_event_data.last_event_age;
					this_ready = current_age >= target_age;
				}
				else if (event->event_type == HSA_EVENTTYPE_HW_EXCEPTION ||
						 event->event_type == HSA_EVENTTYPE_NODECHANGE ||
						 event->event_type == HSA_EVENTTYPE_DEVICESTATECHANGE ||
						 event->event_type == HSA_EVENTTYPE_HW_EXCEPTION ||
						 event->event_type == HSA_EVENTTYPE_DEBUG_EVENT ||
						 event->event_type == HSA_EVENTTYPE_PROFILE_EVENT ||
						 event->event_type == HSA_EVENTTYPE_MEMORY)
				{
					// These never happen in the model
				}
				else
				{
					fprintf(stderr, "model: Unimplemented event type\n");
					abort();
				}
				if (final_ready != this_ready)
				{
					final_ready = this_ready;
					break;
				}
			}
			if (final_ready)
				break;
			if (have_timeout)
			{
				int ret = pthread_cond_timedwait(
					&model_event_condvar, &model_ioctl_mutex, &timeout);
				if (ret == ETIMEDOUT)
				{
					hit_timeout = true;
					break;
				}
			}
			else
			{
				pthread_cond_wait(&model_event_condvar, &model_ioctl_mutex);
			}
		}
		/* Record most recent event ages and perform auto reset. */
		for (unsigned i = 0; i < args->num_events; ++i)
		{
			unsigned slot = events[i].event_id - 1;
			struct model_event *event = &model_events[slot];
			if (event->event_type == HSA_EVENTTYPE_SIGNAL)
			{
				uint64_t last_age = event->value;
				if (event->auto_reset && last_age >= events[i].signal_event_data.last_event_age)
					event->value = 0;
				events[i].signal_event_data.last_event_age = last_age;
			}
		}
		args->wait_result = hit_timeout ? KFD_IOC_WAIT_RESULT_TIMEOUT
										: KFD_IOC_WAIT_RESULT_COMPLETE;
		return 0;
	}
	case AMDKFD_IOC_SET_EVENT:
	{
		struct kfd_ioctl_set_event_args *args = arg;
		model_set_event(NULL, args->event_id);
		return 0;
	}
	case AMDKFD_IOC_RESET_EVENT:
	{
		pr_debug("MODEL IOCTL: AMDKFD_IOC_RESET_EVENT\n");
		struct kfd_ioctl_reset_event_args *args = arg;
		unsigned slot = args->event_id - 1;
		struct model_event *event = &model_events[slot];
		if (event->event_type == HSA_EVENTTYPE_SIGNAL)
		{
			model_events[slot].value = 0;
		}
		else
		{
			fprintf(stderr, "model: Unimplemented event type\n");
			abort();
		}
		return 0;
	}
	case AMDKFD_IOC_DESTROY_EVENT:
	{
		struct kfd_ioctl_destroy_event_args *args = arg;
		unsigned i = args->event_id - 1;
		if (i >= model_event_limit || !(model_event_bitmap[i / 64] & ((uint64_t)1 << (i % 64))))
		{
			fprintf(stderr, "model: trying to destroy an event that doesn't exist.\n");
			abort();
		}
		memset(&model_events[i], 0, sizeof(model_events[i]));
		model_event_bitmap[i / 64] &= ~((uint64_t)1 << (i % 64));
		return 0;
	}
	case AMDKFD_IOC_CREATE_QUEUE:
	{
		pr_debug("MODEL IOCTL: AMDKFD_IOC_CREATE_QUEUE\n");
		struct kfd_ioctl_create_queue_args *args = arg;
		unsigned node_id = args->gpu_id - 1;
		assert(node_id < model_num_nodes);
		assert(model_nodes[node_id].model);
		const bool supported_queue_type = args->queue_type == KFD_IOC_QUEUE_TYPE_COMPUTE_AQL ||
										  args->queue_type == KFD_IOC_QUEUE_TYPE_SDMA;
		if (!supported_queue_type)
		{
			fprintf(stderr, "model: Unsupported queue type\n");
			abort();
		}
		unsigned queue_id = 0;
		while (queue_id < MAX_MODEL_QUEUES && model_queues[queue_id].queue)
			queue_id++;
		if (queue_id >= MAX_MODEL_QUEUES)
		{
			fprintf(stderr, "model: too many queues\n");
			abort();
		}
		struct hsakmt_model_queue_info info = {0};
		info.ring_base_address = args->ring_base_address;
		info.ring_size = args->ring_size;
		info.write_pointer_address = args->write_pointer_address;
		info.read_pointer_address = args->read_pointer_address;
		info.queue_type = args->queue_type;
		model_queues[queue_id].queue =
			model_functions->register_queue(model_nodes[node_id].model, &info);
		model_queues[queue_id].node_id = node_id;
		args->queue_id = queue_id;
		// Note that strictly speaking, this is the offset into the hsakmt_kfd_fd
		// file, not the DRM fd (but they are the same in our case).
		args->doorbell_offset = model_nodes[node_id].doorbell_offset + 8 * queue_id;
		return 0;
	}
	case AMDKFD_IOC_DESTROY_QUEUE:
	{
		struct kfd_ioctl_destroy_queue_args *args = arg;
		if (args->queue_id >= MAX_MODEL_QUEUES || !model_queues[args->queue_id].queue)
		{
			fprintf(stderr, "model: trying to destroy a queue that doesn't exist\n");
			abort();
		}
		struct model_queue *queue = &model_queues[args->queue_id];
		// Older model versions simply leak the queue.
		if (model_functions->version_minor >= 3)
			model_functions->destroy_queue(model_nodes[queue->node_id].model, queue->queue);
		queue->queue = NULL;
		return 0;
	}
	case AMDKFD_IOC_GET_TILE_CONFIG:
	{
		pr_debug("MODEL IOCTL: AMDKFD_IOC_GET_TILE_CONFIG\n");
		struct kfd_ioctl_get_tile_config_args *args = arg;
		args->gb_addr_config = 0x10000444;
		return 0;
	}
	case AMDKFD_IOC_SET_SCRATCH_BACKING_VA:
		pr_debug("MODEL IOCTL: AMDKFD_IOC_SET_SCRATCH_BACKING_VA\n");
		// no-op -- scratch allocations are communicated via amd_queue_s
		return 0;
	case AMDKFD_IOC_RUNTIME_ENABLE:
		pr_debug("MODEL IOCTL: AMDKFD_IOC_RUNTIME_ENABLE\n");
		fprintf(stderr, "model: Debugger runtime not implemented\n");
		fprintf(stderr, "Fix this by clearing bit 30 of the 'capability' field in $HSA_MODEL_TOPOLOGY/%%d/properties\n");
		abort();
	default:
		fprintf(stderr, "model: Unimplemented KFD ioctl\n");
		abort();
	}
}
int model_kfd_ioctl(unsigned long request, void *arg)
{
	/* Use a very simle locking strategy for correctness. IOCTLs should
	 * be rare anyway and not contended considering the cost of running
	 * the model itself.
	 *
	 * The bulk of model execution happens in a separate thread *without*
	 * holding the IOCTL mutex. */
	pthread_mutex_lock(&model_ioctl_mutex);
	int ret = model_kfd_ioctl_locked(request, arg);
	pthread_mutex_unlock(&model_ioctl_mutex);
	return ret;
}