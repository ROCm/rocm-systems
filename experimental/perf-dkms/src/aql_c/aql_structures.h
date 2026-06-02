/**
 * @file aql_structures.h
 * @brief C header file with structures similar to Rust rocprofiler-oop implementation
 */

#ifndef AQL_STRUCTURES_H
#define AQL_STRUCTURES_H

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/stddef.h>
#else
#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
/* Userspace equivalent of kernel atomic_t */
typedef atomic_int atomic_t;
/* Userspace atomic operations */
#define atomic_set(ptr, val) atomic_store(ptr, val)
#define atomic_read(ptr) atomic_load(ptr)
#endif

#include "pm4_packets.h"

/* Forward declarations */
struct kvec_u32;
struct kvec_counter_reg_info;
typedef struct counter_reg_info counter_reg_info_t;
typedef struct block_delay_info block_delay_info_t;
typedef struct dimension dimension_t;

/* Enumerations */
typedef enum {
	HARDWARE_DIM_XCC,
	HARDWARE_DIM_SE,
	HARDWARE_DIM_SA,
	HARDWARE_DIM_CU,
	HARDWARE_DIM_WGP
} hardware_dimensions_t;

typedef enum {
	HW_IP_BLOCK_ATC, /* Address Translation Cache */
	HW_IP_BLOCK_CPC, /* Command Processor Compute */
	HW_IP_BLOCK_CPF, /* Command Processor Fetcher */
	HW_IP_BLOCK_CPG, /* Command Processor Graphics */
	HW_IP_BLOCK_DB, /* Depth Buffer */
	HW_IP_BLOCK_EA, /* Efficiency Arbiter */
	HW_IP_BLOCK_GDS, /* Global Data Store */
	HW_IP_BLOCK_GRBM, /* Graphics Register Bus Manager */
	HW_IP_BLOCK_GL2C, /* Graphics L2 Cache C */
	HW_IP_BLOCK_PA_SC, /* Primitive Assembly - Scan Converter */
	HW_IP_BLOCK_PA_SU, /* Primitive Assembly - Setup Unit */
	HW_IP_BLOCK_RMI, /* Render Backend Memory Interface */
	HW_IP_BLOCK_SPI, /* Shader Processor Input */
	HW_IP_BLOCK_SQ, /* Shader Processor (Sequencer) */
	HW_IP_BLOCK_SX, /* Shader Export */
	HW_IP_BLOCK_TA, /* Texture Addresser */
	HW_IP_BLOCK_TCP, /* Texture Cache Processor */
	HW_IP_BLOCK_TD, /* Texture Data */
	HW_IP_BLOCK_TCA, /* Texture Cache Arbiter */
	HW_IP_BLOCK_TCC, /* Texture Cache Controller */
	HW_IP_BLOCK_UMC, /* Unified Memory Controller */
	HW_IP_BLOCK_SDMA,
	HW_IP_BLOCK_LAST
} hardware_ip_block_t;

/* Architecture type enumeration */
typedef enum { ARCH_TYPE_GFX9, ARCH_TYPE_GFX10, ARCH_TYPE_GFX11, ARCH_TYPE_GFX12 } arch_type_t;

/* Block info - represents a hardware performance counter block type */
typedef struct {
	const char *name; /* Unique string identifier */
	hardware_ip_block_t id; /* Block ID */
	uint32_t instance_count; /* Maximum number of block instances per shader array */
	uint32_t event_id_max; /* Maximum counter event ID */
	uint32_t counter_count; /* Maximum number of counters that can be enabled at once */
	counter_reg_info_t *counter_reg_info; /* Array of counter register addresses */
	uint32_t attr; /* Block attributes mask */
	block_delay_info_t *delay_info; /* Optional block delay info */
	uint32_t spm_block_id; /* SPM block id */

	/* Hardware dimensions for this block */
	dimension_t *dimensions; /* Array of dimensions relevant to this block */
	uint32_t dimension_count; /* Number of dimensions for this block */
} block_info_t;

/* Block info map - provides fast lookup by hardware_ip_block_t */
typedef struct {
	/* Direct indexed array for O(1) lookup by block ID */
	block_info_t *blocks[HW_IP_BLOCK_LAST];

	/* Total number of blocks defined */
	uint32_t block_count;
} block_info_map_t;

/* Architecture-specific control registers for packet generation */
typedef struct {
	/* Global control registers */
	uint32_t grbm_gfx_index; /* GRBM graphics index register */
	uint32_t cp_perfmon_cntl; /* CP performance monitor control */
	uint32_t compute_perfcount_enable; /* Compute performance counter enable */
	uint32_t sq_perfcounter_ctrl2; /* SQ performance counter control 2 */
	uint32_t sq_perfcounter_ctrl;      /* GFX9: SQ_PERFCOUNTER_CTRL (0 on GFX10+) */
	uint32_t sq_perfcounter_mask;      /* GFX9: SQ_PERFCOUNTER_MASK (0 on GFX10+) */
	uint32_t rlc_perfmon_clk_cntl;     /* GFX9: RLC_PERFMON_CLK_CNTL (0 on GFX10+) */

	/* Register space bases */
	uint32_t uconfig_space_start; /* Base of UCONFIG register space */
	uint32_t persistent_space_start; /* Base of persistent (SH) register space */

	/* Event configuration */
	uint32_t cs_partial_flush_event; /* Event type for CS partial flush */
	uint32_t event_index_flush; /* Event index for flush operations */

	/* Cache coherency parameters */
	uint32_t gcr_cntl_default; /* Default GCR control value */
	uint32_t cp_coher_cntl_default;    /* GFX9: CP_COHER_CNTL value for ACQUIRE_MEM (0 on GFX10+) */
	uint32_t poll_interval_default; /* Default poll interval for cache ops */

	/* Performance counter control bits */
	struct {
		uint8_t sq_ps_en_bit; /* Bit position for PS enable */
		uint8_t sq_gs_en_bit; /* Bit position for GS enable */
		uint8_t sq_hs_en_bit; /* Bit position for HS enable */
		uint8_t sq_cs_en_bit; /* Bit position for CS enable */
		uint8_t sq_vs_en_bit;   /* GFX9: VS enable (bit 1), 0 on GFX10+ */
		uint8_t sq_es_en_bit;   /* GFX9: ES enable (bit 3), 0 on GFX10+ */
		uint8_t sq_ls_en_bit;   /* GFX9: LS enable (bit 5), 0 on GFX10+ */
	} counter_control_bits;

	/* GFX9 SQ select value masks (for constructing SQ_PERFCOUNTER_SELECT) */
	struct {
		uint32_t simd_mask;             /* 0xF on GFX9, 0 on GFX12 */
		uint32_t sqc_bank_mask;         /* 0xF on GFX9, 0 on GFX12 */
		uint32_t sqc_client_mask;       /* 0xF on GFX9, 0 on GFX12 */
		uint8_t simd_mask_shift;        /* 24 on GFX9 */
		uint8_t sqc_bank_mask_shift;    /* 12 on GFX9 */
		uint8_t sqc_client_mask_shift;  /* 16 on GFX9 */
	} sq_select_masks;

	/* Perfmon states */
	struct {
		uint8_t perfmon_state_disable; /* Disable state value */
		uint8_t perfmon_state_enable; /* Enable state value */
		uint8_t perfmon_state_stop; /* Stop state value */
		uint8_t perfmon_sample_bit; /* Sample enable bit position */
	} perfmon_states;
} arch_control_regs_t;

/* Forward declare arch_event_map_t from counter_registry.h */
typedef struct arch_event_map arch_event_map_t;

/* Unified architecture structure - same fields for all architectures */
typedef struct {
	arch_type_t type;
	uint32_t num_xcc;
	uint32_t num_se;
	uint32_t num_sa;
	uint32_t num_cu;
	uint32_t num_wgp_per_sa;

	/* PM4 command buffer for building GPU command streams */
	pm4_buffer_t *command;

	/* Block info organized by hardware IP block type */
	block_info_map_t block_map;

	/* Architecture-specific control registers */
	arch_control_regs_t control_regs;

	/* Architecture-specific event mapping */
	const arch_event_map_t *event_map;
	size_t event_count;
} arch_t;

/* Counter allocation state */
typedef enum {
	COUNTER_STATE_FREE = 0,
	COUNTER_STATE_ALLOCATED,
	COUNTER_STATE_RESERVED, /* Reserved by system */
	COUNTER_STATE_ERROR
} counter_state_t;

/* Counter allocation info - tracks what a counter is being used for */
typedef struct {
	atomic_t state; /* Atomic state for lock-free allocation */
	uint32_t event_id; /* Event being monitored */
	uint32_t instance_id; /* Which instance of the block (0 to instance_count-1) */
	uint32_t user_id; /* User-defined ID for tracking */
	const char *description; /* Optional description of what's being measured */
	uint64_t allocation_time; /* Timestamp when allocated */

	/* GPU data buffer pointers (provided by AQL queue manager) */
	void     *data_cpu_addr;  /* Kernel VA for reading counter results */
	uint64_t  data_gpu_addr;  /* GPU VA for COPY_DATA destination */
	uint32_t  data_size;      /* Size of available data buffer */
} counter_allocation_t;

/* Counter register information with allocation tracking */
struct counter_reg_info {
	/* Hardware register addresses */
	uint32_t select_addr;
	uint32_t control_addr;
	uint32_t register_addr_lo;
	uint32_t register_addr_hi;

	/* Allocation tracking */
	counter_allocation_t allocation;
};

/* Block delay information */
struct block_delay_info {
	uint32_t reg;
	uint32_t val;
};

/* Hardware dimension information */
struct dimension {
	uint32_t size;
	hardware_dimensions_t dim;
};

/* GPU block information */
typedef struct {
	const char *name;
	hardware_ip_block_t id;
	uint32_t instance_count;
	uint32_t event_id_max;
	uint32_t counter_count;
	struct kvec_counter_reg_info *counter_reg_info;
	uint32_t attr;
	block_delay_info_t *delay_info; /* Optional - NULL if not present */
	uint32_t spm_block_id;
} gpu_block_info_t;

/* Counter descriptor */
typedef struct {
	const char *name;
	hardware_ip_block_t id;
	uint32_t event_id;
} counter_t;

/* PM4 command wrapper */
typedef struct {
	uint32_t value;
} pm4_command_t;

/* Bitfield structures for GFX12 */

/* CP Perfmon Control */
typedef union {
	uint32_t raw;
	struct {
		uint32_t perfmon_state : 4; /* bits 0-3 */
		uint32_t spm_perfmon_state : 4; /* bits 4-7 */
		uint32_t perfmon_enable_mode : 2; /* bits 8-9 */
		uint32_t perfmon_sample_enable : 1; /* bit 10 */
		uint32_t reserved : 21; /* bits 11-31 */
	} bits;
} cp_perfmon_cntl_t;

/* GRBM GFX Index */
typedef union {
	uint32_t raw;
	struct {
		uint32_t instance_index : 7; /* bits 0-6 */
		uint32_t reserved1 : 1; /* bit 7 */
		uint32_t sa_index : 2; /* bits 8-9 */
		uint32_t reserved2 : 6; /* bits 10-15 */
		uint32_t se_index : 4; /* bits 16-19 */
		uint32_t reserved3 : 9; /* bits 20-28 */
		uint32_t sa_broadcast_writes : 1; /* bit 29 */
		uint32_t instance_broadcast_writes : 1; /* bit 30 */
		uint32_t se_broadcast_writes : 1; /* bit 31 */
	} bits;
} grbm_gfx_index_t;

/* Barrier Event */
typedef union {
	uint32_t raw;
	struct {
		uint32_t event_type : 6; /* bits 0-5 */
		uint32_t reserved1 : 2; /* bits 6-7 */
		uint32_t event_index : 4; /* bits 8-11 */
		uint32_t reserved2 : 9; /* bits 12-20 */
		uint32_t samp_plst_cntr_mode : 2; /* bits 21-22 */
		uint32_t offload_enable : 1; /* bit 23 */
		uint32_t reserved3 : 8; /* bits 24-31 */
	} bits;
} barrier_event_t;

/* Copy Data */
typedef union {
	uint32_t raw;
	struct {
		uint32_t src_sel : 4; /* bits 0-3 */
		uint32_t reserved1 : 4; /* bits 4-7 */
		uint32_t dst_sel : 4; /* bits 8-11 */
		uint32_t reserved2 : 1; /* bit 12 */
		uint32_t src_temporal : 2; /* bits 13-14 */
		uint32_t reserved3 : 1; /* bit 15 */
		uint32_t count_sel : 1; /* bit 16 */
		uint32_t reserved4 : 3; /* bits 17-19 */
		uint32_t wr_confirm : 1; /* bit 20 */
		uint32_t mode : 1; /* bit 21 */
		uint32_t reserved5 : 1; /* bit 22 */
		uint32_t aid_id : 2; /* bits 23-24 */
		uint32_t dst_temporal : 2; /* bits 25-26 */
		uint32_t reserved6 : 2; /* bits 27-28 */
		uint32_t pq_exe_status : 1; /* bit 29 */
		uint32_t reserved7 : 2; /* bits 30-31 */
	} bits;
} copy_data_t;

/* Performance Counter Control 2 */
typedef union {
	uint32_t raw;
	struct {
		uint32_t force_en : 1; /* bit 0 */
		uint32_t vmid_en : 16; /* bits 1-16 */
		uint32_t reserved : 15; /* bits 17-31 */
	} bits;
} perf_counter_ctrl2_t;

/* SQ Performance Control */
typedef union {
	uint32_t raw;
	struct {
		uint32_t ps_en : 1; /* bit 0 */
		uint32_t reserved1 : 1; /* bit 1 */
		uint32_t gs_en : 1; /* bit 2 */
		uint32_t reserved2 : 1; /* bit 3 */
		uint32_t hs_en : 1; /* bit 4 */
		uint32_t reserved3 : 1; /* bit 5 */
		uint32_t cs_en : 1; /* bit 6 */
		uint32_t reserved4 : 7; /* bits 7-13 */
		uint32_t disable_me0pipe0 : 1; /* bit 14 */
		uint32_t disable_me0pipe1 : 1; /* bit 15 */
		uint32_t disable_me1pipe0 : 1; /* bit 16 */
		uint32_t disable_me1pipe1 : 1; /* bit 17 */
		uint32_t disable_me1pipe3 : 1; /* bit 18 */
		uint32_t reserved5 : 13; /* bits 19-31 */
	} bits;
} sq_perf_control_t;

/* Performance Counter Select */
typedef union {
	uint32_t raw;
	struct {
		uint32_t perf_sel : 9; /* bits 0-8 */
		uint32_t reserved1 : 11; /* bits 9-19 */
		uint32_t spm_mode : 4; /* bits 20-23 */
		uint32_t reserved2 : 4; /* bits 24-27 */
		uint32_t perf_mode : 4; /* bits 28-31 */
	} bits;
} perf_counter_select_t;

/* Compute Performance Count Enable */
typedef union {
	uint32_t raw;
	struct {
		uint32_t enable : 1; /* bit 0 */
		uint32_t reserved : 31; /* bits 1-31 */
	} bits;
} compute_perfcount_enable_t;

/* KFD Memory Flags */
typedef union {
	uint32_t raw;
	struct {
		uint32_t non_paged : 1; /* bit 0 */
		uint32_t cache_policy : 2; /* bits 1-2 */
		uint32_t read_only : 1; /* bit 3 */
		uint32_t page_size : 2; /* bits 4-5 */
		uint32_t host_access : 1; /* bit 6 */
		uint32_t no_substitute : 1; /* bit 7 */
		uint32_t gds_memory : 1; /* bit 8 */
		uint32_t scratch : 1; /* bit 9 */
		uint32_t atomic_access_full : 1; /* bit 10 */
		uint32_t atomic_access_partial : 1; /* bit 11 */
		uint32_t execute_access : 1; /* bit 12 */
		uint32_t coarse_grain : 1; /* bit 13 */
		uint32_t aql_queue_memory : 1; /* bit 14 */
		uint32_t fixed_address : 1; /* bit 15 */
		uint32_t no_numa_bind : 1; /* bit 16 */
		uint32_t uncached : 1; /* bit 17 */
		uint32_t no_address : 1; /* bit 18 */
		uint32_t only_address : 1; /* bit 19 */
		uint32_t extended_coherent : 1; /* bit 20 */
		uint32_t gtt_access : 1; /* bit 21 */
		uint32_t contiguous : 1; /* bit 22 */
		uint32_t reserved : 9; /* bits 23-31 */
	} bits;
} kfd_memory_flags_t;

/* PM4 Packet Structures */

/* Set UConfig Register packet */
typedef struct {
	uint32_t header; /* Contains op_code and size */
	uint16_t reg_offset;
	uint16_t reserved;
	uint32_t reg_value;
} set_uconfig_reg_packet_t;

/* Event Write packet */
typedef struct {
	uint32_t header; /* Contains op_code and size */
	uint32_t event;
} event_write_packet_t;

/* Copy Data packet */
typedef struct {
	uint32_t header; /* Contains op_code and size */
	uint32_t copy_data;
	uint32_t src_reg_offset_lo;
	uint32_t src_reg_offset_hi;
	uint32_t dst_reg_offset_lo;
	uint32_t dst_reg_offset_hi;
} copy_data_packet_t;

/* Flush Cache packet */
typedef struct {
	uint32_t header; /* Contains op_code and size */
	uint32_t reserved;
	uint32_t coher_size;
	uint32_t coher_size_hi;
	uint32_t coher_base_lo;
	uint32_t coher_base_hi;
	uint32_t poll_interval;
	uint32_t gcr_cntl;
} flush_cache_packet_t;

/* Write SH Register packet */
typedef struct {
	uint32_t header; /* Contains op_code and size */
	union {
		uint32_t raw;
		struct {
			uint32_t reg_offset : 16;
			uint32_t reserved : 7;
			uint32_t vmid_shift : 5;
			uint32_t index : 4;
		} fields;
	} word1;
	uint32_t reg_value;
} write_sh_register_packet_t;

/* Helper macros for PM4 packet creation */
#define PM4_HEADER(op_code, size_dw) (((op_code) << 8) | (((size_dw) - 2) & 0x3FFF) << 16)

#define SET_UCONFIG_REG_OPCODE 0x79
#define EVENT_WRITE_OPCODE 0x46 /* decimal 70 */
#define COPY_DATA_OPCODE 0x40 /* decimal 64 */
#define FLUSH_CACHE_OPCODE 0x58
#define WRITE_SH_REG_OPCODE 0x76

/* Allocation summary structure */
typedef struct {
	uint32_t total_counters;
	uint32_t free_counters;
	uint32_t allocated_counters;
	uint32_t reserved_counters;
	uint32_t error_counters;
} counter_allocation_summary_t;

/* Function pointer type for block iteration */
typedef void (*block_info_visitor_fn)(hardware_ip_block_t block_id, block_info_t *block,
				      void *user_data);

/* Helper macros for counter allocation */
#define COUNTER_ALLOCATE_SIMPLE(block, event_id, instance_id) \
	counter_allocate(block, event_id, instance_id, 0, NULL)

#define COUNTER_IS_FREE(counter) (counter_get_state(counter) == COUNTER_STATE_FREE)

#define COUNTER_IS_ALLOCATED(counter) (counter_get_state(counter) == COUNTER_STATE_ALLOCATED)

/* All function declarations commented out for now */
/*
// Block info map access functions
void block_info_map_init(block_info_map_t* map);
void block_info_map_add(block_info_map_t* map, block_info_t* block);
block_info_t* block_info_map_get(block_info_map_t* map, hardware_ip_block_t block_id);
block_info_t* block_info_map_find_by_name(block_info_map_t* map, const char* name);
void block_info_map_foreach(block_info_map_t* map, block_info_visitor_fn visitor, void* user_data);
uint32_t block_info_map_find_by_attr(block_info_map_t* map, uint32_t attr_mask, block_info_t** results, uint32_t max_results);
uint32_t block_info_map_get_total_counters(block_info_map_t* map);
uint32_t block_info_map_get_total_instances(block_info_map_t* map, uint32_t num_shader_arrays);

// Counter allocation management functions
counter_reg_info_t* counter_allocate(block_info_t* block, uint32_t event_id, uint32_t instance_id, uint32_t user_id, const char* description);
bool counter_free(counter_reg_info_t* counter);
counter_reg_info_t* counter_find_free(block_info_t* block);
counter_state_t counter_get_state(const counter_reg_info_t* counter);
bool counter_is_available(const counter_reg_info_t* counter);
const counter_allocation_t* counter_get_allocation(const counter_reg_info_t* counter);
bool counter_reserve(counter_reg_info_t* counter, const char* reason);
uint32_t block_info_get_free_counter_count(const block_info_t* block);
uint32_t block_info_get_allocated_counter_count(const block_info_t* block);
uint32_t block_info_get_counters_by_state(const block_info_t* block, counter_state_t state, counter_reg_info_t** results, uint32_t max_results);
void block_info_reset_counter_allocations(block_info_t* block);
counter_reg_info_t* counter_find_by_event(block_info_t* block, uint32_t event_id, uint32_t instance_id);
counter_reg_info_t* counter_find_by_user_id(block_info_t* block, uint32_t user_id);

// Architecture-level counter management functions
uint32_t arch_get_total_free_counters(arch_t* arch);
uint32_t arch_get_total_allocated_counters(arch_t* arch);
counter_reg_info_t* arch_find_counter_by_user_id(arch_t* arch, uint32_t user_id, hardware_ip_block_t* out_block_id);
void arch_reset_all_counter_allocations(arch_t* arch);
void arch_get_allocation_summary(arch_t* arch, counter_allocation_summary_t* summary);

// Debug and reporting functions
void counter_debug_print_block_status(const block_info_t* block);
void counter_debug_print_arch_status(const arch_t* arch);
const char* counter_state_to_string(counter_state_t state);

// Builder function declarations
counter_reg_info_t* counter_reg_info_new(void);
counter_reg_info_t* counter_reg_info_select_addr(counter_reg_info_t* self, uint32_t addr);
counter_reg_info_t* counter_reg_info_control_addr(counter_reg_info_t* self, uint32_t addr);
counter_reg_info_t* counter_reg_info_register_addr_lo(counter_reg_info_t* self, uint32_t addr);
counter_reg_info_t* counter_reg_info_register_addr_hi(counter_reg_info_t* self, uint32_t addr);
block_delay_info_t* block_delay_info_new(void);
block_delay_info_t* block_delay_info_reg(block_delay_info_t* self, uint32_t reg);
block_delay_info_t* block_delay_info_val(block_delay_info_t* self, uint32_t val);
gpu_block_info_t* gpu_block_info_new(void);
gpu_block_info_t* gpu_block_info_name(gpu_block_info_t* self, const char* name);
gpu_block_info_t* gpu_block_info_id(gpu_block_info_t* self, hardware_ip_block_t id);
gpu_block_info_t* gpu_block_info_instance_count(gpu_block_info_t* self, uint32_t count);
gpu_block_info_t* gpu_block_info_event_id_max(gpu_block_info_t* self, uint32_t max);
gpu_block_info_t* gpu_block_info_counter_count(gpu_block_info_t* self, uint32_t count);
gpu_block_info_t* gpu_block_info_counter_reg_info(gpu_block_info_t* self, struct kvec_counter_reg_info* info);
gpu_block_info_t* gpu_block_info_attr(gpu_block_info_t* self, uint32_t attr);
gpu_block_info_t* gpu_block_info_delay_info(gpu_block_info_t* self, block_delay_info_t* delay);
gpu_block_info_t* gpu_block_info_spm_block_id(gpu_block_info_t* self, uint32_t id);
counter_t* counter_new(const char* name, hardware_ip_block_t id, uint32_t event_id);
const char* counter_get_name(const counter_t* self);
hardware_ip_block_t counter_get_id(const counter_t* self);
uint32_t counter_get_event_id(const counter_t* self);
const char* gpu_block_info_get_name(const gpu_block_info_t* self);
hardware_ip_block_t gpu_block_info_get_id(const gpu_block_info_t* self);
uint32_t gpu_block_info_get_instance_count(const gpu_block_info_t* self);
uint32_t gpu_block_info_get_event_id_max(const gpu_block_info_t* self);
uint32_t gpu_block_info_get_counter_count(const gpu_block_info_t* self);
const struct kvec_counter_reg_info* gpu_block_info_get_counter_reg_info(const gpu_block_info_t* self);
uint32_t gpu_block_info_get_attr(const gpu_block_info_t* self);
const block_delay_info_t* gpu_block_info_get_delay_info(const gpu_block_info_t* self);
uint32_t gpu_block_info_get_spm_block_id(const gpu_block_info_t* self);
*/

/* pm4_op_t and related types are now defined in pm4_packets.h */

/* Architecture creation and destruction functions */
arch_t *arch_create_by_name(const char *arch_name);
void arch_destroy(arch_t *arch);

#endif /* AQL_STRUCTURES_H */