/*
 * aql_perf.h - AQL Performance Counter Integration Header
 *
 * This header defines the AQL packet submission system for performance
 * counter integration in the perf-dkms kernel module.
 */

#ifndef _AQL_PERF_H
#define _AQL_PERF_H

#include <linux/types.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>
#include <linux/refcount.h>
#include <linux/list.h>
#include <linux/atomic.h>
#include <linux/workqueue.h>
#include <linux/ktime.h>
#include <linux/perf_event.h>
#include "aql_c/aql_structures.h"
#include "aql_c/packet_generation.h"
#include "pmu_dimension.h"

/* Forward declarations */
struct file;
struct kfd_process;
struct amdgpu_ib;

/* KFD Data Allocation Structure - from KFD module */
struct kfd_data_alloc {
	void *cpu_addr;
	void *gpu_addr;
	struct amdgpu_ib *ib;
	size_t size;
};

/* AQL Performance Counter Constants */
#define AQL_PERF_MAX_GPUS 16
#define AQL_PERF_RESULT_BUFFER_SIZE (PAGE_SIZE)
#define AQL_PERF_PACKET_BUFFER_SIZE (PAGE_SIZE)
#define AQL_PERF_MAX_RECOVERY_ATTEMPTS 3

/* GFX12 Counter Constants */
#define GFX12_PERF_SEL_SQ_WAVES 0x00000001
#define GFX12_PERF_SEL_SQ_INSTS 0x00000002
#define GFX12_PERF_SEL_TA_BUSY 0x00000003
#define GFX12_GPU_TYPE 12

/* Error Codes */
#define AQL_PERF_SUCCESS 0
#define AQL_PERF_ERR_NO_DEVICE -ENODEV
#define AQL_PERF_ERR_NO_MEMORY -ENOMEM
#define AQL_PERF_ERR_INVALID -EINVAL
#define AQL_PERF_ERR_TIMEOUT -ETIMEDOUT
#define AQL_PERF_ERR_GPU_FAULT -EIO

/* Forward declaration for module parameter */
extern bool debug_enable;

#define aql_debug(fmt, ...)                                                 \
	do {                                                                \
		if (debug_enable)                                           \
			printk(KERN_DEBUG "AQL_PERF: " fmt, ##__VA_ARGS__); \
	} while (0)

#define aql_info(fmt, ...) printk(KERN_INFO "AQL_PERF: " fmt, ##__VA_ARGS__)

#define aql_warn(fmt, ...) printk(KERN_WARNING "AQL_PERF: WARNING: " fmt, ##__VA_ARGS__)

#define aql_err(fmt, ...) printk(KERN_ERR "AQL_PERF: ERROR: " fmt, ##__VA_ARGS__)

/* Session States */
enum session_state {
	SESSION_UNINITIALIZED = 0,
	SESSION_INITIALIZING,
	SESSION_ACTIVE,
	SESSION_MEASURING,
	SESSION_ERROR,
	SESSION_DESTROYING
};

/* Measurement States */
enum measurement_state {
	MEASUREMENT_IDLE = 0,
	MEASUREMENT_STARTING,
	MEASUREMENT_ACTIVE,
	MEASUREMENT_READING,
	MEASUREMENT_STOPPING,
	MEASUREMENT_ERROR
};

/* Work Operation Types */
enum aql_work_op_type { AQL_WORK_START = 0, AQL_WORK_STOP, AQL_WORK_READ };

/* AQL Work Structure for Deferred Operations */
struct aql_work_item {
	struct work_struct work;
	struct aql_measurement *measurement;
	enum aql_work_op_type op_type;
	struct completion *completion; /* For synchronous operations */
	int result; /* Operation result */
};

/* AQL Packet Types */
enum aql_perf_packet_type {
	AQL_PERF_PACKET_START = 0x01,
	AQL_PERF_PACKET_READ = 0x02,
	AQL_PERF_PACKET_END = 0x03
};

/* Error Severity Levels */
enum aql_error_severity {
	AQL_ERROR_RECOVERABLE = 0,
	AQL_ERROR_GPU_FAULT,
	AQL_ERROR_SYSTEM_FAULT,
	AQL_ERROR_PERMANENT
};

/* Statistics Types */
enum aql_stat_type {
	AQL_STAT_PACKETS_SUBMITTED = 0,
	AQL_STAT_PACKETS_COMPLETED,
	AQL_STAT_ERRORS_TOTAL,
	AQL_STAT_SESSIONS_CREATED
};

/* Performance Statistics */
struct aql_perf_stats {
	atomic64_t packets_submitted;
	atomic64_t packets_completed;
	atomic64_t errors_total;
	atomic64_t sessions_created;
};

/* Error Context */
struct aql_error_context {
	enum aql_error_severity severity;
	uint32_t gpu_id;
	int error_code;
	char error_msg[128];
	ktime_t timestamp;
};

/* Error Recovery Structure */
struct aql_error_recovery {
	unsigned long error_mask; /* Bitmap of GPU errors */
	struct delayed_work recovery_work; /* Delayed work for recovery */
	int recovery_attempts; /* Track recovery attempts */
};

/* Counter Descriptor */
struct counter_descriptor {
	uint32_t counter_id;
	uint32_t counter_select;
	uint32_t counter_mode;
	const char *name;
	const char *description;
	uint32_t supported_gpus;
};

/* GFX12 Counter Configuration */
struct gfx12_counter_desc {
	uint32_t counter_select;
	uint32_t counter_mode;
	uint64_t *result_buffer;
	size_t result_size;
};

/* Counter Configuration */
struct counter_config {
	uint32_t num_counters;
	uint32_t max_counters;
	struct gfx12_counter_desc *descriptors;
	uint64_t *counter_masks;
};

/* AQL Packet Structure */
struct aql_perf_packet {
	uint32_t packet_type;
	uint32_t gpu_id;
	uint64_t counter_select;
	uint64_t result_address;
	uint32_t packet_data[16];
	size_t packet_size;
};

/* Forward declarations */
typedef struct counter_reg_info counter_reg_info_t;
struct pmu_dimension_coords;
struct shared_counter_ref;

/* Shared counter tracking for dimension-specific events */
struct shared_counter_ref {
	uint32_t counter_id; /* Which counter type (counter_id_t) */
	struct aql_measurement *measurement; /* The measurement that owns the counter */
	atomic_t ref_count; /* How many events share this counter */
	struct list_head list; /* List linkage */
};

/* Per-measurement tracking structure */
struct aql_measurement {
	struct list_head list;
	struct aql_perf_session *session;
	uint32_t gpu_id;
	uint32_t counter_mask;
	uint32_t counter_id; /* Counter ID from counter_registry (counter_id_t) */
	struct perf_event *event;
	ktime_t start_time;
	enum measurement_state state;
	uint64_t start_counter_value; /* Counter value when measurement started */
	uint64_t last_counter_value; /* Most recent counter value read */

	/* Allocated counter from block (NULL if not allocated) */
	counter_reg_info_t *allocated_counter;

	/* Counter sharing support */
	struct shared_counter_ref *shared_ref; /* Shared counter reference */
	bool owns_counter; /* True if this measurement allocated the counter */

	/* Dimension-specific monitoring support */
	struct pmu_dimension_coords target_dims; /* Target hardware dimensions */
	bool dimension_specific; /* True if targeting specific dimensions */

	/* Work queue support for atomic context handling */
	spinlock_t cache_lock; /* Protects cached_counter_value */
	uint64_t cached_counter_value; /* Cached value for atomic reads */
	bool cache_valid; /* Whether cached value is valid */
	bool pending_destroy; /* Measurement should be freed after async work */

	/* Reference counting to prevent use-after-free with work items */
	struct kref refcount; /* Reference count for safe async operations */
};

/* Main AQL Performance Session Structure */
struct aql_perf_session {
	/* Core KFD integration */
	struct file *kfd_file;
	struct kfd_process *process;

	/* GPU management */
	uint32_t *gpu_ids;
	uint32_t num_gpus;
	uint32_t max_gpus;

	/* Thread safety and state management */
	struct mutex session_mutex;
	refcount_t ref_count; /* Reference counting for future multi-session support.
                                          * Currently only one global session exists, but this
                                          * infrastructure allows safe sharing if needed. */
	enum session_state state;

	/* Resource tracking */
	struct list_head active_measurements;
	spinlock_t measurement_lock;
	atomic_t active_gpu_count;

	/* Counter sharing for dimension-specific events */
	struct list_head shared_counters; /* List of shared counter allocations */
	spinlock_t shared_lock; /* Protects shared_counters list */

	/* Error recovery */
	struct aql_error_recovery recovery;

	/* Performance monitoring */
	struct aql_perf_stats stats;

	/* Counter configuration */
	struct counter_config counters;

	/* GPU Architecture - one per GPU */
	arch_t **archs; /* Array of architecture pointers, indexed by GPU index */

	/* Session ID for debugging */
	uint64_t session_id;
};

/* Function prototypes */

/* Global Workqueue Access */
struct workqueue_struct *aql_get_global_workqueue(void);

/* Session Management */
struct aql_perf_session *aql_perf_session_create(void);
int aql_perf_session_initialize(struct aql_perf_session *session);
void aql_perf_session_destroy(struct aql_perf_session *session);
void aql_perf_session_get(struct aql_perf_session *session);
void aql_perf_session_put(struct aql_perf_session *session);

/* GPU Discovery and Setup */
int aql_perf_discover_gpus(struct aql_perf_session *session);

/* Counter Buffer Management */
int aql_perf_allocate_counter_buffers(arch_t *arch, struct file *kfd_file,
				      struct kfd_process *process, uint32_t gpu_id);
void aql_perf_free_counter_buffers(arch_t *arch, struct file *kfd_file, struct kfd_process *process,
				   uint32_t gpu_id);

/* Counter Allocation Functions */
counter_reg_info_t *aql_counter_try_allocate(block_info_t *block, uint32_t event_id,
					     struct perf_event *perf_event);
void aql_counter_release(counter_reg_info_t *reg);
void release_shared_counter(struct aql_perf_session *session, struct shared_counter_ref *ref);
int aql_build_counter_info(uint32_t counter_id, arch_t *arch, counter_reg_info_t *allocated_counter,
			   counter_info_t *out_info, block_info_t **out_block);

/* Packet Operations - New PM4-based implementation */
int aql_perf_create_start_packet(struct aql_measurement *measurement,
				 pm4_buffer_t **out_pm4_buffer);
int aql_perf_create_read_packet(struct aql_measurement *measurement, pm4_buffer_t **out_pm4_buffer);
int aql_perf_create_end_packet(struct aql_measurement *measurement, pm4_buffer_t **out_pm4_buffer);
int aql_perf_submit_pm4_packet(struct aql_perf_session *session, uint32_t gpu_id,
			       pm4_buffer_t *pm4_buffer);

/* Measurement Management */
struct aql_measurement *aql_perf_measurement_create(struct aql_perf_session *session,
						    uint32_t gpu_id, struct perf_event *event);
int aql_perf_measurement_start(struct aql_measurement *measurement);
int aql_perf_measurement_stop(struct aql_measurement *measurement);
uint64_t aql_perf_measurement_read(struct aql_measurement *measurement);
void aql_perf_measurement_destroy(struct aql_measurement *measurement);

/* Measurement Reference Counting */
void aql_measurement_get(struct aql_measurement *m);
void aql_measurement_put(struct aql_measurement *m);

/* Atomic Context Support - New Functions */
int aql_perf_measurement_start_atomic(struct aql_measurement *measurement);
int aql_perf_measurement_stop_atomic(struct aql_measurement *measurement);
uint64_t aql_perf_measurement_read_atomic(struct aql_measurement *measurement);

/* Work Handler Functions */
void aql_work_handler(struct work_struct *work);
struct aql_work_item *aql_create_work_item(struct aql_measurement *measurement,
					   enum aql_work_op_type op_type);

/* Error Handling */
void aql_perf_handle_error(struct aql_perf_session *session, struct aql_error_context *error);
void aql_perf_recovery_work(struct work_struct *work);

/* Statistics */
void aql_perf_inc_stat(enum aql_stat_type type);
void aql_perf_get_stats(struct aql_perf_stats *stats);

/* Error Recovery Functions */
struct aql_error_context *aql_perf_create_error_context(enum aql_error_severity severity,
							uint32_t gpu_id, int error_code,
							const char *error_msg, ...);
void aql_perf_destroy_error_context(struct aql_error_context *error);
bool aql_perf_is_gpu_disabled(struct aql_perf_session *session, uint32_t gpu_id);
uint32_t aql_perf_get_healthy_gpu_count(struct aql_perf_session *session);

/* PMU Integration Functions */
struct aql_perf_session *aql_pmu_get_session(void);
void aql_pmu_put_session(struct aql_perf_session *session);

#endif /* _AQL_PERF_H */